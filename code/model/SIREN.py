import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# sin activation
class Sine(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x)

# siren layer
class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim = dim_in
        self.is_first = is_first
        self.w0 = w0
        self.c = c

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.activation = Sine() if activation is None else activation

        self.init_()



    def init_(self):
        '''
         In general, the initialization of the first layer is dependent on the frequencies of the signal - 
         higher frequencies require larger weights in the first layer.
        '''
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.dim,
                                            1 / self.dim)
            else:
                self.linear.weight.uniform_(-np.sqrt(self.c / self.dim) / self.w0,
                                            np.sqrt(self.c / self.dim) / self.w0)

    def forward(self, x):
        out =  self.activation(self.linear(self.w0 * x))
        return out


#siren network
class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, w0 = 30., w0_first = 30., use_bias = True, final_activation = None):
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

        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = Siren(dim_in = dims[-2], dim_out = dim_out, w0 = w0, use_bias = use_bias, activation=final_activation)

        # print(self)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        out = self.last_layer(x)

        return out


# modulatory feed forward

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)
