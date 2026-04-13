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


def pad_with_indicator_channels(
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
    - side : padding side in ["prepend", "append"].
    - length : padding length.
    - value : padding value for indicator channel.
    :return:
    """
    n_dims = x.dim()

    # --- validation ---
    if not -n_dims <= channel_dim < n_dims:
        raise ValueError(f"channel_dim ({channel_dim}) outside of input dims.")

    # --- build padding arrays ---
    prepend_paddings = [0] * n_dims
    append_paddings = [0] * n_dims

    for dim, side, length, value in paddings:
        if side == "prepend":
            if prepend_paddings[dim] > 0:
                raise ValueError(f"Duplicate padding configuration for prepend side of dim {dim}.")
            prepend_paddings[dim] = length
        elif side == "append":
            if append_paddings[dim] > 0:
                raise ValueError(f"Duplicate padding configuration for append side of dim {dim}.")
            append_paddings[dim] = length
        else:
            raise ValueError(f"side ({side}) must be one of 'prepend' or 'append'.")

    if prepend_paddings[channel_dim] > 0 or append_paddings[channel_dim] > 0:
        raise ValueError(f"channel_dim ({channel_dim}) in paddings list.")
    prepend_paddings[channel_dim] = len(paddings)

    # --- torch padding format ---
    torch_padding = []
    for lp, hp in zip(reversed(prepend_paddings), reversed(append_paddings)):
        torch_padding.extend([lp, hp])

    padded = F.pad(x, torch_padding)

    # --- fill indicator channels ---
    for i, (dim, side, length, value) in enumerate(paddings):
        pad_slice = [slice(None)] * n_dims
        pad_slice[channel_dim] = i

        if side == "prepend":
            pad_slice[dim] = slice(length)
        else:
            pad_slice[dim] = slice(-length, None)

        padded[*pad_slice] = value

    return padded


def add_constant_channel(x, channel_dim, constant_value=1.0, location="prepend"):
    if location not in ["prepend", "append"]:
        raise ValueError(f"location {location} is not supported (supported : prepend, append).")
    augmentation_size = list(x.size())
    augmentation_size[channel_dim] = 1
    augmentation = torch.empty(augmentation_size, dtype=x.dtype, device=x.device)
    augmentation.fill_(constant_value)
    if location == "prepend":
        x_augmented = torch.cat((augmentation, x), dim=channel_dim)
    else:
        x_augmented = torch.cat((x, augmentation), dim=channel_dim)
    return x_augmented


def crelu(x, dim=-1):
    """
    Concatenated ReLU from
    "Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units"
    """
    return torch.cat([F.relu(x), F.relu(-x)], dim=dim)


def cat_pool(x, channel_dim, spatial_dims, kernel_size):
    # Validation
    dim_usage = [0] * x.dim()
    dim_usage[channel_dim] += 1
    for dim in spatial_dims:
        dim_usage[dim] += 1
    if max(dim_usage) > 1:
        raise ValueError(f"Redondant dimension in [{channel_dim}], {spatial_dims} for torch of size {x.size()}.")

    if len(kernel_size) != len(spatial_dims):
        raise ValueError(f"Kernel size and spatial dims do not match ({len(kernel_size)} != {len(spatial_dims)}).")
    for kernel_dim_size, spatial_dim in zip(kernel_size, spatial_dims):
        if x.size(spatial_dim) % kernel_dim_size != 0:
            raise ValueError(f"Tensor size ({x.size()}) must be divisible by kernel size ({kernel_size}).")

    # 1. Split each spatial dimension into [reduced_size, kernel_size]
    split_dims = [[dim_size] for dim_size in x.size()]
    for kernel_dim_size, spatial_dim in zip(kernel_size, spatial_dims):
        split_dims[spatial_dim] = [split_dims[spatial_dim][0] // kernel_dim_size, kernel_dim_size]

    # 2. Reshape: introduce kernel dimensions as explicit dimensions
    x_split = x.reshape([dim_size for dim_sizes in split_dims for dim_size in dim_sizes])

    # 3. Build mapping: original dim -> indices in x_split
    i = 0
    orig_to_split_indices = []
    for dim_sizes in split_dims:
        orig_to_split_indices.append(list(range(i, i + len(dim_sizes))))
        i += len(dim_sizes)

    # 4. Move kernel indices just before the channel dim
    new_channel_dim_split = [orig_to_split_indices[spatial_dim].pop() for spatial_dim in spatial_dims]
    new_channel_dim_split.append(orig_to_split_indices[channel_dim][0])
    orig_to_split_indices[channel_dim] = new_channel_dim_split

    # 5. Permute to group channel and kernel indices together
    perm = [idx for dim_indices in orig_to_split_indices for idx in dim_indices]
    x_permuted = x_split.permute(perm)

    # 6. Final reshape: merge kernel dimensions into the channel dimension
    final_shape = list(x.size())
    for kernel_dim_size, spatial_dim in zip(kernel_size, spatial_dims):
        final_shape[spatial_dim] //= kernel_dim_size
        final_shape[channel_dim] *= kernel_dim_size

    y = x_permuted.reshape(final_shape)

    return y
