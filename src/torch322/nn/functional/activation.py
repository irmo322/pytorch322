import torch
from torch.nn import functional as F


def crelu(x, dim=-1):
    """
    Concatenated ReLU from
    "Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units"
    """
    return torch.cat([F.relu(x), F.relu(-x)], dim=dim)
