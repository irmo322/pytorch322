# adapted from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py

import math

from torch.nn import Linear
from torch.linalg import matrix_norm


class LinearStdWeight(Linear):
    # weight are constrained: sum(w[i,j]**2) == number of element in weight
    # bias are initialized using standard normal (mu=0, sigma=1)
    # self.normalize_weight() must be called after each weight update in order to constrain weight.

    def __init__(self, *args, use_norm_factor=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_norm_factor = use_norm_factor
        self.norm_factor = 1 / math.sqrt(self.in_features + (0 if self.bias is None else 1))

    def reset_parameters(self):
        self.weight.data.normal_()
        if self.bias is not None:
            self.bias.data.normal_()
        self.normalize_weight()

    def forward(self, *args, **kwargs):
        y = super().forward(*args, **kwargs)
        if self.use_norm_factor:
            y *= self.norm_factor
        return y

    def normalize_weight(self):
        self.weight.data *= math.sqrt(self.weight.data.numel()) / matrix_norm(self.weight.data)
