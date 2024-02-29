from torch import nn
import torch

from consts import NULL_CHK
from typing import Literal

def get_activation(act: str) -> nn.Module:
    return {
        "relu" : nn.ReLU,
        "gelu" : nn.GELU,
    }[act]()

class MLP(nn.Module):
    def __init__(self, activation: Literal['relu', 'gelu'] = "relu", dimensions: list = [2,2,2]):
        super(MLP, self).__init__()

        layers = [ ]
        last_dim = dimensions[0]
        for dim in dimensions[1:]:
            layers.append(
                nn.Linear(last_dim, dim)
            )

            last_dim = dim
            layers.append(
                get_activation(activation)
            )
        layers = layers[:-1] # remove the extra activation

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLPSequence(nn.Module):

    def __init__(self, n_dims=1, context_len=1, activation = 'relu', hidden_dimensions: list = [8, 8], **kwargs):

        super(MLPSequence, self).__init__()

        NULL_CHK(n_dims, context_len, hidden_dimensions, activation)

        self.name = f"MLP_ctx{context_len}_" + "_".join(map(str, hidden_dimensions))
        self.sequence_model = True

        y_dim = 1
        x_dim = n_dims
        
        self.n_dims = x_dim
        self.context_len = context_len
        
        # input dimension of current_x + context_len * (x_dim + y_dim)
        dimensions = [x_dim + context_len * (x_dim + y_dim)] \
                        + hidden_dimensions \
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
