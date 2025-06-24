import torch

def get_beta(a: torch.Tensor):
    return torch.norm(a, p=1) / a.numel()

def get_gamma(a: torch.Tensor):
    return torch.max(torch.abs(a))

def calculate_alpha(tensor: torch.Tensor):
  return torch.mean(tensor)

def get_qb_range(b: int): 
    return -2 ** (b -1), 2 ** (b-1)