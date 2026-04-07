# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html

import math

from torch.nn import Conv2d, ConvTranspose2d
from torch.linalg import vector_norm


class Conv2dStdWeight(Conv2d):
    # change reset parameters using normal with mu=0 and std=1
    # self.normalize_weight() must be called after each weight update in order to constrain weight.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_factor = 1 / math.sqrt(self.in_channels // self.groups * self.kernel_size[0] * self.kernel_size[1]
                                         + (0 if self.bias is None else 1))

    def reset_parameters(self):
        self.weight.data.normal_()
        if self.bias is not None:
            self.bias.data.normal_()
        self.normalize_weight()

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs) * self.norm_factor

    def normalize_weight(self):
        self.weight.data *= math.sqrt(self.weight.data.numel()) / vector_norm(self.weight.data)


class ConvTranspose2dStdWeight(ConvTranspose2d):
    # change reset parameters using normal with mu=0 and std=1
    # self.normalize_weight() must be called after each weight update in order to constrain weight.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_factor = 1 / math.sqrt(self.in_channels // self.groups * self.kernel_size[0] * self.kernel_size[1]
                                         / (self.stride[0] * self.stride[1]) + (0 if self.bias is None else 1))

    def reset_parameters(self):
        self.weight.data.normal_()
        if self.bias is not None:
            self.bias.data.normal_()
        self.normalize_weight()

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs) * self.norm_factor

    def normalize_weight(self):
        self.weight.data *= math.sqrt(self.weight.data.numel()) / vector_norm(self.weight.data)
