import math

import torch

from torch322.nn.functional import add_constant_channel


class ChannelNorm(torch.nn.Module):
    """Applies channel normalization on input tensor.

    Values are normalized along the channel dimension.
    Optionally, a constant channel can be added.

    Args:
        channel_dim: The dimension along which channels are normalized.
        constant_channel: If True, a constant channel will be added.
        constant_channel_value: The value of the added channel.
        constant_channel_location: The location of the added channel, "prepend" or "append".
    """
    def __init__(
            self,
            channel_dim,
            constant_channel=False,
            constant_channel_value=1.0,
            constant_channel_location="prepend",
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.constant_channel = constant_channel
        self.constant_channel_value = constant_channel_value
        self.constant_channel_location = constant_channel_location

    def forward(self, x):
        if self.constant_channel:
            x_augmented = add_constant_channel(
                x,
                self.channel_dim,
                constant_value=self.constant_channel_value,
                location=self.constant_channel_location
            )
        else:
            x_augmented = x

        n_channels = x_augmented.size()[self.channel_dim]
        normalisation_tensor = math.sqrt(n_channels) * torch.rsqrt((x_augmented * x_augmented).sum(dim=self.channel_dim, keepdim=True))
        x_normalized = x_augmented * normalisation_tensor

        return x_normalized
