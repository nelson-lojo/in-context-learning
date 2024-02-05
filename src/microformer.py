import torch
from torch import nn


DTYPE = torch.float32
DEVICE= "cpu"

def count_parameters(model: nn.Module):
    params_by_layer = [
        torch.prod(torch.tensor(p.shape)) 
        for p in model.parameters()
    ]
    return sum(params_by_layer)

def get_causal_mask(length: int):
    return nn.Transformer.generate_square_subsequent_mask(length)

class MicroFormer(nn.Module):

    def __init__(self, n_dims, n_embd, out_dim = 1, n_layer=1, n_head=1, ff_dims=None, **kwargs):
        super(MicroFormer, self).__init__()

        in_dim = n_dims
        internal_dim = n_embd
        internal_layers = n_layer

        self.n_dims = in_dim

        if ff_dims is None:
            ff_dims = [internal_dim, internal_dim, internal_dim]

        self.expansion = nn.Linear(in_dim, internal_dim, dtype=DTYPE)
        self.tsf = nn.Sequential(*[
            MicroFormerLayer(internal_dim, num_heads=n_head, ff_dims=ff_dims)
            for _ in range(internal_layers)
        ])
        self.reduction = nn.Linear(internal_dim, out_dim, dtype=DTYPE)
    
    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        combined = self._combine(xs, ys)
        ac = self.expansion(combined)
        ac = self.tsf(ac) # this should return the activation head at each element in the sequence
        ac = self.reduction(ac)
        return ac[:, ::2, 0][:, inds]  # predict only on xs

class MicroFormerLayer(nn.Module):

    def __init__(self, dimension: int, num_heads: int = 1, ff_dims=None):
        super(MicroFormerLayer, self).__init__()

        # self.residual_weight_1 = nn.parameter.Parameter(torch.tensor([0]))
        # self.residual_weight_2 = nn.parameter.Parameter(torch.tensor([0]))

        assert dimension % num_heads == 0, f"Embedding dimension {dimension} is not evenly divisible over {num_heads} head(s)!"

        if ff_dims is None:
            ff_dims = [dimension, dimension, dimension]

        # RMSLayerNorm is what's used in llama2
        self.pre_attn_norm = RMSLayerNorm()
        self.pre_attn_map = nn.Linear(dimension, dimension)
        # for a large number of heads, there is always GroupedQueryAttention: https://arxiv.org/pdf/2305.13245.pdf
        self.attn = nn.MultiheadAttention(dimension, num_heads, batch_first=True) 
        self.pre_mlp_norm = RMSLayerNorm()
        self.mlp = MLP(activation=(lambda: nn.ReLU()), dimensions=ff_dims)

    def forward(self, x):
        # should return the activation head at all tokens 
        # x is of shape (bs, seq_len, embed_dim), out is same shape
        bs, seq_len, embed_dim = x.shape

        skip1 = x
        activation = self.pre_attn_norm(skip1)
        activation = self.pre_attn_map(activation)
        # this call below is based on <>._sa_block() visible at https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
        activation, _ = self.attn(activation, activation, activation, need_weights=False, attn_mask=get_causal_mask(seq_len))#, is_causal=True) # when upgrading to torch 2.0, replace attn_mask=... with is_causal=...
        # also I don't know why <MultiheadAttention>.forward() returns a tuple with None in the only other (second) slot
        activation += skip1

        skip2 = activation
        activation = self.pre_mlp_norm(skip2)
        activation = self.mlp(activation)
        return activation + skip2


# this may be replaced by nn.MultiheadAttention
"""
class MyAttention(nn.Module):
    def __init__(self, dim, qk_dim=None, heads=1):
        super(MyAttention, self).__init__()
        if qk_dim is None:
            qk_dim = dim

        self.qk_dim = qk_dim
        self.query = [ nn.Linear(dim, qk_dim) for _ in range(heads) ]
        self.key   = [ nn.Linear(dim, qk_dim) for _ in range(heads) ]
        self.value = [ nn.Linear(dim, dim)    for _ in range(heads) ]

    def forward(self, x):
        b_size, seq_len, embed_dim = x.shape
        # x is the normalized input embedding
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        assert query.shape == (b_size, seq_len, self.qk_dim)
        assert key.shape   == (b_size, seq_len, self.qk_dim)
        assert value.shape == (b_size, seq_len, embed_dim)

        ## TODO: this line may be optimized to halve the # of matmuls
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) # Q * K.T
        assert attn_weights.shape == (b_size, seq_len, seq_len)
        attn_weights = attn_weights / (value.size(-1)**0.5) # divide by sqrt(d)
        assert attn_weights.shape == (b_size, seq_len, seq_len)

        # TODO: where are the heads coming in?
        if causal := True:
            mask = get_causal_mask(seq_len)
            attn_weights = torch.matmul(attn_weights, mask)
            ## TODO: check dims
        
        result = torch.matmul(attn_weights, value)

        return result#, attn_weights ## TODO: why do they also return the attn_weights?
"""
        
class RMSLayerNorm(nn.Module):
    def __init__(self):
        super(RMSLayerNorm, self).__init__()

    def forward(self, x):
        item_values = x.reshape(x.shape[0], -1)
        aggregate = torch.sum(item_values ** 2, axis=1) 
        aggregate /= item_values.shape[1]
        aggregate = torch.sqrt(aggregate)
        assert aggregate.shape == (x.shape[0], )

        scaled = (item_values.T / aggregate).T

        return scaled.reshape(x.shape)


class MLP(nn.Module):
    def __init__(self, activation=(lambda: nn.ReLU()), dimensions: list = [2,2,2]):
        super(MLP, self).__init__()

        layers = [ ]
        last_dim = dimensions[0]
        for dim in dimensions[1:]:
            layers.append(
                nn.Linear(last_dim, dim, dtype=DTYPE)
            )

            last_dim = dim
            layers.append(
                activation()
            )
        layers = layers[:-1] # remove the extra activation

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

