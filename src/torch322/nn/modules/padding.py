from torch import nn

from ..functional import pad322, pad_with_indicator_channels


class Padder322(nn.Module):
    
    def __init__(self, channel_dim, spatial_dims, spatial_paddings, padding_values):
        super().__init__()
        self.channel_dim = channel_dim
        self.spatial_dims = spatial_dims
        self.spatial_paddings = spatial_paddings
        self.padding_values = padding_values
    
    def forward(self, x):
        return pad322(x, self.channel_dim, self.spatial_dims, self.spatial_paddings, self.padding_values)


class PadderWithIndicatorChannels(nn.Module):

    def __init__(self, channel_dim, paddings):
        super().__init__()
        self.channel_dim = channel_dim
        self.paddings = paddings

    def forward(self, x):
        return pad_with_indicator_channels(x, self.channel_dim, self.paddings)
