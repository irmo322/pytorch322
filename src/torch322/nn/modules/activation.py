from torch import nn

from ..functional import crelu


class CReLU(nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return crelu(x, self.dim)
