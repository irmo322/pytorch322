# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html

import math

import torch
from torch.linalg import vector_norm
from torch.nn import Conv2d, ConvTranspose2d

import torch322


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


class ConvSparseKernel(torch.nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: torch.Size,
            kernel_keys,
            stride: tuple[int, ...] = None,
            bias: bool = True,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # args check
        self.dim = len(kernel_size)
        for key in kernel_keys:
            if len(key) != self.dim:
                raise ValueError(f"Key {key} in kernel_keys should have {self.dim} elements.")
            if not all(0 <= key_comp < dim_size for key_comp, dim_size in zip(key, kernel_size, strict=True)):
                raise ValueError(f"Key {key} is out of kernel size {kernel_size}.")
        if stride is not None and len(stride) != self.dim:
            raise ValueError(f"Stride {stride} should have {self.dim} elements.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_keys = kernel_keys
        self.stride = stride if stride is not None else [1] * self.dim

        self.lins = torch.nn.ModuleList([
            torch322.nn.LinearStdWeight(self.in_channels, self.out_channels, use_norm_factor=False, bias=False)
            for _ in self.kernel_keys
        ])

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.norm_factor = 1 / math.sqrt(self.in_channels * len(self.kernel_keys) + (0 if self.bias is None else 1))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.bias is not None:
            self.bias.data.normal_()

    def forward(self, x):
        # expect input x of size (*B, in_channels, *S)
        # where :
        # - *B : batch, or whatever
        # - *S : spatial dimensions (len(S) == len(kernel_size))

        input_size = x.size()
        spatial_input_size = input_size[-self.dim:]

        tsc = torch322.utils.TensorSizeChecker()
        tsc.check(x, ["*B", self.in_channels, *spatial_input_size])

        spatial_output_size_no_stride = torch.Size([
            i_d - k_d + 1
            for i_d, k_d in zip(spatial_input_size, self.kernel_size, strict=True)
        ])
        spatial_output_size_with_stride = torch.Size([
            (i_d - k_d) // s_d + 1
            for i_d, k_d, s_d in zip(spatial_input_size, self.kernel_size, self.stride, strict=True)
        ])

        if min(spatial_output_size_no_stride) <= 0:
            raise ValueError(f"Spatial input size {spatial_input_size} is too small for kernel size {self.kernel_size}.")

        x_channel_last = torch.movedim(x, -self.dim - 1, -1)
        tsc.check(x_channel_last, ["*B", *spatial_input_size, self.in_channels])

        lin_outputs_sum = None
        for key, lin in zip(self.kernel_keys, self.lins, strict=True):
            lin_input = x_channel_last[
                ...,
                *[
                    slice(k_d, k_d + s_d * o_d, s_d)
                    for k_d, o_d, s_d in zip(key, spatial_output_size_with_stride, self.stride, strict=True)
                ],
                slice(None)
            ]
            tsc.check(lin_input, ["*B", *spatial_output_size_with_stride, self.in_channels])

            lin_output = lin(lin_input)
            tsc.check(lin_output, ["*B", *spatial_output_size_with_stride, self.out_channels])

            lin_outputs_sum = lin_output if lin_outputs_sum is None else lin_outputs_sum + lin_output

        # add bias
        y_channel_last = lin_outputs_sum if self.bias is None else lin_outputs_sum + self.bias
        tsc.check(y_channel_last, ["*B", *spatial_output_size_with_stride, self.out_channels])

        # normalize
        y_channel_last_normalized = y_channel_last * self.norm_factor

        y = torch.movedim(y_channel_last_normalized, -1, -self.dim - 1)
        tsc.check(y, ["*B", self.out_channels, *spatial_output_size_with_stride])

        return y
