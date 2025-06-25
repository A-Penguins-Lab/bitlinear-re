import torch
import torch.nn as nn

from utils import get_qb_range, get_beta, get_gamma

class ABSMax:
    def __init__(self, gamma: float, beta: float, ep: float, bit_range: int = 8):
        
        self.gamma = gamma
        self.beta = beta
        self.ep = ep

        self.qb_min, self.qb_max = get_qb_range(bit_range)
        self.p,self.q = self.qb_min + ep, self.qb_max - ep

    def _abs_clip(self, tensor):
        return torch.maximum(torch.tensor(self.p), torch.minimum(torch.tensor(self.q), tensor))

    def quantize(self, x: torch.Tensor, cast: bool = False):
        x = x * (self.qb_max / self.gamma)
        x = self._abs_clip(x)

        if cast:return x.to(torch.int8)
        else: return x

class BitLinear(nn.Module):
    
    def __init__(self, n_in: int, n_out: int, ep: float, bit_range: int = 8):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.ep = ep
        self.bit = bit_range
        self.ln = nn.LayerNorm(n_in)

        self.qb_max, _  = get_qb_range(self.bit)
        self.weight_init()

    def weight_init(self):
        self.w = torch.randn(self.n_in, self.n_out)

        self.alpha = torch.mean(self.w)
        self.beta = get_beta(self.w)
        self.gamma = get_gamma(self.w)

        self.q = ABSMax(
            self.gamma,
            self.beta,
            self.ep,
            self.bit
        )

        self.w =self.w - self.alpha

        ## Signum fn applying
        self.w = torch.where(self.w > 0, 1.0, -1.0)
        self.w = nn.Parameter(self.w, requires_grad=True)

    def forward(self, x):
        x = self.ln(x)
        x = self.q.quantize(x)
    
        final_term = (self.beta * self.gamma) / self.qb_max
        y = x @ self.w * final_term

        

        return y

class FreezeBitLinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward():
        pass        
