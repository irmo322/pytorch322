from torch.nn.functional import pad


def pad322(
        input_,
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
    :param input_: input tensor. Any size is accepted.
    :param channel_dim: index of the channel dimension.
    :param spatial_dims: indexes of the spatial dimensions. list of integers.
    :param spatial_paddings: left and right paddings for each spatial dimension.
    :param padding_values: left and right padding values for each spatial dimension.
    :return:
    """

    n_dims = input_.dim()
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

    padded = pad(input_, torch_padding)

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
