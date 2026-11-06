import torch
from torch import nn
from .conv_layer import ConvLayer

class SnakeLinearized(nn.Module):
    def __init__(self, features, module=None, alpha=None, beta=None):
        super().__init__()
        self.features = features
        if module is not None:
            alpha = module.alpha if not module.logscale else torch.exp(module.alpha)
            beta = module.beta if not module.logscale else torch.exp(module.beta)
        self.register_buffer(
            "alpha", 
            torch.zeros(1, features, 1) if alpha is None 
            else alpha.detach().clone()
        )
        self.register_buffer(
            "beta", 
            torch.zeros(1, features, 1) if beta is None 
            else beta.detach().clone()
        )
        # Initialize as identity
        self.weight = torch.nn.Parameter(torch.ones(1, features, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, features, 1))

    def original(self, x):
        return torch.sin(self.alpha * x) ** 2 / self.beta

    def forward(self, x):
        return x * self.weight + self.bias

    def __matmul__(self, other: ConvLayer):
        weight, bias = other._combine_conv1d(
            other.weight, other.bias,
            torch.diag(self.weight.detach()).unsqueeze(0), self.bias.detach()
        )
        return weight, bias