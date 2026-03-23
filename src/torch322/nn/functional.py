import torch

from torch.nn import functional as F


def pad322(
        x,
        channel_dim,
        spatial_dims,
        spatial_paddings,
        padding_values,
):
    """
    Pad spatial dimensions and prepend indicator channels encoding padding regions.

    One channel is added per spatial dimension. In each indicator channel,
    the left and right padded regions of the corresponding spatial dimension
    are filled with the provided padding values.
    :param x: input tensor. Any size is accepted.
    :param channel_dim: index of the channel dimension.
    :param spatial_dims: indexes of the spatial dimensions. list of integers.
    :param spatial_paddings: left and right paddings for each spatial dimension.
    :param padding_values: left and right padding values for each spatial dimension.
    :return:
    """

    n_dims = x.dim()
    n_spatial = len(spatial_dims)

    # --- validation ---
    if not 0 <= channel_dim < n_dims:
        raise ValueError(f"channel_dim ({channel_dim}) outside of input dims.")

    if not all(0 <= d < n_dims for d in spatial_dims):
        raise ValueError(f"spatial_dims ({spatial_dims}) outside of input dims.")

    if channel_dim in spatial_dims:
        raise ValueError(f"channel_dim ({channel_dim}) and spatial_dims ({spatial_dims}) are mutually exclusive.")

    if len(spatial_paddings) != n_spatial:
        raise ValueError(
            f"spatial_dims and spatial_paddings must have same length ({len(spatial_dims)} != {len(spatial_paddings)}).")

    if len(padding_values) != n_spatial:
        raise ValueError(
            f"spatial_dims and padding_values must have same length ({len(spatial_dims)} != {len(padding_values)}).")

    # --- build padding arrays ---
    left_paddings = [0] * n_dims
    right_paddings = [0] * n_dims

    for dim, (lp, rp) in zip(spatial_dims, spatial_paddings, strict=True):
        left_paddings[dim] = lp
        right_paddings[dim] = rp

    # --- torch padding format ---
    torch_padding = []
    for lp, rp in zip(reversed(left_paddings), reversed(right_paddings)):
        torch_padding.extend([lp, rp])

    torch_padding[-2 * (channel_dim + 1)] += n_spatial

    padded = F.pad(x, torch_padding)

    # --- fill indicator channels ---
    for i, (dim, (lp, rp), (lv, rv)) in enumerate(zip(
            spatial_dims, spatial_paddings, padding_values, strict=True)
    ):
        pad_slice = [slice(None)] * n_dims
        pad_slice[channel_dim] = i

        if lp > 0:
            pad_slice[dim] = slice(lp)
            padded[*pad_slice] = lv

        if rp > 0:
            pad_slice[dim] = slice(-rp, None)
            padded[*pad_slice] = rv

    return padded


def pad323(
        x,
        channel_dim,
        paddings,
):
    """
    Pad dimensions and prepend indicator channels encoding padding regions.
    :param x: input tensor. Any size is accepted.
    :param channel_dim: channel dimension.
    :param paddings: list of (dim, side, length, value) where
    - dim : padding dimension.
    - side : padding side in ["low", "high"].
    - length : padding length.
    - value : padding value for indicator channel.
    :return:
    """
    n_dims = x.dim()

    # --- validation ---
    if not -n_dims <= channel_dim < n_dims:
        raise ValueError(f"channel_dim ({channel_dim}) outside of input dims.")

    # --- build padding arrays ---
    low_paddings = [0] * n_dims
    high_paddings = [0] * n_dims

    for dim, side, length, value in paddings:
        if side == "low":
            if low_paddings[dim] > 0:
                raise ValueError(f"Duplicate padding configuration for low side of dim {dim}.")
            low_paddings[dim] = length
        elif side == "high":
            if high_paddings[dim] > 0:
                raise ValueError(f"Duplicate padding configuration for high side of dim {dim}.")
            high_paddings[dim] = length
        else:
            raise ValueError(f"side ({side}) must be one of 'low' or 'high'.")

    if low_paddings[channel_dim] > 0 or high_paddings[channel_dim] > 0:
        raise ValueError(f"channel_dim ({channel_dim}) in paddings list.")
    low_paddings[channel_dim] = len(paddings)

    # --- torch padding format ---
    torch_padding = []
    for lp, hp in zip(reversed(low_paddings), reversed(high_paddings)):
        torch_padding.extend([lp, hp])

    padded = F.pad(x, torch_padding)

    # --- fill indicator channels ---
    for i, (dim, side, length, value) in enumerate(paddings):
        pad_slice = [slice(None)] * n_dims
        pad_slice[channel_dim] = i

        if side == "low":
            pad_slice[dim] = slice(length)
        else:
            pad_slice[dim] = slice(-length, None)

        padded[*pad_slice] = value

    return padded


def crelu(x, dim=-1):
    """
    Concatenated ReLU from
    "Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units"
    """
    return torch.cat([F.relu(x), F.relu(-x)], dim=dim)
