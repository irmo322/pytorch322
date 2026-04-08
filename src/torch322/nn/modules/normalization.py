import math

import torch

from torch322.nn.functional import add_constant_channel


class ChannelNorm(torch.nn.Module):
    """Applies channel normalization on input tensor.

    Values are normalized along the channel dimension.
    Optionally, channels can be prepended to integrate normalization information.

    Args:
        channel_dim: The dimension along which channels are normalized.
        constant_channel: an extra channel with constant value is added before normalization.
        constant_channel_value: The value of the added channel.
        factor_log_channel: an extra channel is added after normalization. Its value is the natural log of the
            normalization factor (one for each vector of channels).
    """
    def __init__(
            self,
            channel_dim,
            constant_channel=False,
            constant_channel_value=1.0,
            factor_log_channel=False,
    ):
        super().__init__()

        self.channel_dim = channel_dim
        self.constant_channel = constant_channel
        self.constant_channel_value = constant_channel_value
        self.factor_log_channel = factor_log_channel

    def forward(self, x):
        if self.constant_channel:
            x_with_constant = add_constant_channel(
                x,
                self.channel_dim,
                constant_value=self.constant_channel_value
            )
        else:
            x_with_constant = x

        normalization_tensor = _compute_normalization_tensor(x_with_constant, self.channel_dim)
        x_normalized = x_with_constant * normalization_tensor

        if self.factor_log_channel:
            x_normalized_with_log = torch.cat((normalization_tensor.log(), x_normalized), dim=self.channel_dim)
        else:
            x_normalized_with_log = x_normalized

        return x_normalized_with_log


def _compute_normalization_tensor(x, channel_dim):
    n_channels = x.size()[channel_dim]
    normalization_tensor = math.sqrt(n_channels) * torch.rsqrt((x * x).sum(dim=channel_dim, keepdim=True))
    return normalization_tensor
