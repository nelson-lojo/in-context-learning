from torch import nn
import torch

from typing import Optional


class MLP(nn.Module):
    def __init__(self, activation=(lambda: nn.ReLU()), dimensions: list = [2,2,2]):
        super(MLP, self).__init__()

        layers = [ ]
        last_dim = dimensions[0]
        for dim in dimensions[1:]:
            layers.append(
                nn.Linear(last_dim, dim)
            )

            last_dim = dim
            layers.append(
                activation()
            )
        layers = layers[:-1] # remove the extra activation

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLPSequence(nn.Module):

    def __init__(self, n_dims=1, n_embd=2, n_layer=2, context_len=1, activation=(lambda: nn.ReLU()), dimensions: Optional[list] = None, **kwargs):

        super(MLPSequence, self).__init__()

        self.name = f"MLP_n={context_len}"
        self.sequence_model = True

        y_dim = 1
        x_dim = n_dims
        
        self.n_dims = x_dim
        self.context_len = context_len

        if dimensions is None:
            # input dimension of current_x + context_len * (x_dim + y_dim)
            dimensions = [x_dim + context_len * (x_dim + y_dim)] \
                         + [n_embd] * n_layer \
                         + [1]

        self.net = MLP(activation, dimensions)

    def _combine(self, xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        try:
            y_dim = ys_b.shape[2]
        except IndexError as e:
            y_dim = 1
        
        xy_seq = torch.cat(
            (xs_b, 
             ys_b.view(ys_b.shape + (1,))), 
            axis=2
        )

        contexted = [
            torch.cat((torch.zeros(bsize, i, dim+y_dim), xy_seq[:, :-i,:]), axis=1)
            for i in range(1, self.context_len + 1)
        ]

        return torch.cat(contexted + [xs_b], axis=-1) # returns (b_size, seq_len, x_dim + ctx_len * (x_dim + y_dim))

    def forward(self, xs, ys, inds=None):
        input = self._combine(xs, ys)
        # now input is of shape (b_size, seq_len, x_dim + ctx_len * (x_dim + y_dim))
        out_ys = self.net(input)[..., 0]
        
        return out_ys
