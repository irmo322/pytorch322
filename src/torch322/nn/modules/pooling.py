import torch

from ..functional.cat_pooling import cat_pool, cat_unpool


class CatPool(torch.nn.Module):
    def __init__(self, channel_dim, spatial_dims, kernel_size):
        super().__init__()
        self.channel_dim = channel_dim
        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size

    def forward(self, x):
        return cat_pool(x, self.channel_dim, self.spatial_dims, self.kernel_size)


class CatUnpool(torch.nn.Module):
    def __init__(self, channel_dim, spatial_dims, kernel_size):
        super().__init__()
        self.channel_dim = channel_dim
        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size

    def forward(self, x):
        return cat_unpool(x, self.channel_dim, self.spatial_dims, self.kernel_size)
