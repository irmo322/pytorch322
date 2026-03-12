from torch.nn.functional import pad


def pad322(input_, channel_dim, spatial_dims, spatial_paddings, padding_values):
    """
    Create a padded input with extra channels for indicating padding (one channel for each spatial dim).
    :param input_: input tensor. Any size is accepted.
    :param channel_dim: index of the channel dimension.
    :param spatial_dims: indexes of the spatial dimensions. list of integers.
    :param spatial_paddings: left and right paddings for each spatial dimension.
    :param padding_values: left and right padding values for each spatial dimension.
    :return:
    """

    n_dims = len(input_.size())
    if not 0 <= channel_dim < n_dims:
        raise ValueError(f"channel_dim ({channel_dim}) outside of input dims.")
    if not all(0 <= spatial_dim < n_dims for spatial_dim in spatial_dims):
        raise ValueError(f"spatial_dims ({spatial_dims}) outside of input dims.")
    if channel_dim in spatial_dims:
        raise ValueError(f"channel_dim ({channel_dim}) and spatial_dims ({spatial_dims}) are mutually exclusive.")

    if len(spatial_paddings) != len(spatial_dims):
        raise ValueError(f"lengths of spatial_dims and spatial_paddings do not match ({len(spatial_dims)} != {len(spatial_paddings)}).")
    if len(padding_values) != len(spatial_dims):
        raise ValueError(f"lengths of spatial_dims and padding_values do not match ({len(spatial_dims)} != {len(padding_values)}).")
    
    left_paddings = [0] * n_dims
    right_paddings = [0] * n_dims
    for spatial_dim, (lp, rp) in zip(spatial_dims, spatial_paddings, strict=True):
        left_paddings[spatial_dim] = lp
        right_paddings[spatial_dim] = rp

    new_size = [d + lp + rp for d, lp, rp in zip(input_.size(), left_paddings, right_paddings, strict=True)]
    new_size[channel_dim] += len(spatial_dims)

    torch_padding = []
    for lp, rp in zip(left_paddings[::-1], right_paddings[::-1]):
        torch_padding.extend([lp, rp])
    torch_padding[-2 * (channel_dim + 1)] += len(spatial_dims)

    padded_input = pad(input_, torch_padding)

    for i, (spatial_dim, (lp, rp), (lv, rv)) in enumerate(zip(
            spatial_dims, spatial_paddings, padding_values, strict=True)):
        complete_slice = [slice(None)] * n_dims
        complete_slice[channel_dim] = i

        complete_slice[spatial_dim] = slice(lp)
        padded_input[*complete_slice] = lv

        complete_slice[spatial_dim] = slice(-rp, None)
        padded_input[*complete_slice] = rv

    return padded_input
