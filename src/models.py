import torch
import torch.nn as nn

## training loop
from bitlinear import BitLinear

class SimpleModel(nn.Module):
    def __init__(self, n_in, n_out, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            BitLinear(n_in if i == 0 else n_out, n_out, ep=1e-9, bit_range=8)
            for i in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x

class SimpleLinearModel(nn.Module):
    def __init__(self, n_in, n_out, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(n_in if i == 0 else n_out, n_out)
            for i in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x    


class CNN(nn.Module):
    pass
