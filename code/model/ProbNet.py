from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.SIREN import Siren

class PNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, skip_in=()):
        super().__init__()

        dims = [dim_in] + dim_hidden + [dim_out]
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - dim_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)


            setattr(self, "lin" + str(layer), lin)

        self.activation = nn.Sigmoid()


    def forward(self, input):

        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            # if layer < self.num_layers - 2:
            #     x = self.activation(x)
            
            x = self.activation(x)

        return x


class PSNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, w0 = 1., w0_first = 30., use_bias = True, final_activation = None):
        super().__init__()
        dims = [dim_in] + dim_hidden + [dim_out]
        self.num_layers = len(dims)
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(self.num_layers-2):
            is_first = ind == 0
            layer_w0 = w0_first if is_first else w0
            layer_dim_in = dims[ind]
            layer_dim_out = dims[ind+1]

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = layer_dim_out,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        final_activation = nn.Sigmoid() if final_activation is None else final_activation
        self.last_layer = Siren(dim_in = dims[-2], dim_out = dim_out, w0 = w0, use_bias = use_bias, activation=final_activation)

        # print(self)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        out = self.last_layer(x)

        return out
