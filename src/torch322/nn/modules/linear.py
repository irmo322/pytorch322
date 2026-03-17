# adapted from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
# change reset parameters using normal with mu=0 and std=1
from torch.nn import Linear

import math


class LinearStdWeight(Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_factor = 1 / math.sqrt(self.in_features + (0 if self.bias is None else 1))

    def reset_parameters(self):
        self.weight.data.normal_()
        if self.bias is not None:
            self.bias.data.normal_()

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs) * self.norm_factor
