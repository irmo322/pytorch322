
def cat_pool(x, channel_dim, spatial_dims, block_size):
    """
    Reduce spatial dimensions by folding spatial values into the channel dimension.

    For each spatial dimension, a block window is slid over the dimension and the
    values within each window are concatenated into the channel dimension. This is
    a lossless operation: all information is preserved, unlike average or max pooling.

    This is the inverse of cat_unpool.

    Args:
        x (torch.Tensor): Input tensor of any number of dimensions.
        channel_dim (int): Index of the channel dimension. Supports negative indexing.
        spatial_dims (list[int]): Indices of the spatial dimensions to pool over.
            Supports negative indexing. Must not overlap with channel_dim.
        block_size (list[int]): block size for each spatial dimension. Each spatial
            dimension size must be divisible by the corresponding block size.

    Returns:
        torch.Tensor: Output tensor with the same number of dimensions as x, where
            each spatial dimension is divided by its block size, and the channel
            dimension is multiplied by the product of all block sizes.

    Raises:
        ValueError: If channel_dim and spatial_dims overlap.
        ValueError: If block_size and spatial_dims have different lengths.
        ValueError: If any spatial dimension is not divisible by its block size.

    Example:
        >>> x = torch.randn(10, 3, 4, 4)
        >>> y = cat_pool(x, channel_dim=1, spatial_dims=[2, 3], block_size=[2, 2])
        >>> y.shape
        torch.Size([10, 12, 2, 2])
    """
    # Validation
    dim_usage = [0] * x.dim()
    dim_usage[channel_dim] += 1
    for dim in spatial_dims:
        dim_usage[dim] += 1
    if max(dim_usage) > 1:
        raise ValueError(f"Redundant dimension in [{channel_dim}], {spatial_dims} for torch of size {x.size()}.")

    if len(block_size) != len(spatial_dims):
        raise ValueError(f"block size and spatial dims do not match ({len(block_size)} != {len(spatial_dims)}).")
    for block_dim_size, spatial_dim in zip(block_size, spatial_dims):
        if x.size(spatial_dim) % block_dim_size != 0:
            raise ValueError(f"Tensor size ({x.size()}) must be divisible by block size ({block_size}).")

    # 1. Split each spatial dimension into [reduced_size, block_size]
    split_dims = [[dim_size] for dim_size in x.size()]
    for block_dim_size, spatial_dim in zip(block_size, spatial_dims):
        split_dims[spatial_dim] = [split_dims[spatial_dim][0] // block_dim_size, block_dim_size]

    # 2. Reshape: introduce block dimensions as explicit dimensions
    x_split = x.reshape([dim_size for dim_sizes in split_dims for dim_size in dim_sizes])

    # 3. Build mapping: original dim -> indices in x_split
    i = 0
    orig_to_split_indices = []
    for dim_sizes in split_dims:
        orig_to_split_indices.append(list(range(i, i + len(dim_sizes))))
        i += len(dim_sizes)

    # 4. Move block indices just before the channel dim
    new_channel_dim_split = [orig_to_split_indices[spatial_dim].pop() for spatial_dim in spatial_dims]
    new_channel_dim_split.append(orig_to_split_indices[channel_dim][0])
    orig_to_split_indices[channel_dim] = new_channel_dim_split

    # 5. Permute to group channel and block indices together
    perm = [idx for dim_indices in orig_to_split_indices for idx in dim_indices]
    x_permuted = x_split.permute(perm)

    # 6. Final reshape: merge block dimensions into the channel dimension
    final_shape = list(x.size())
    for block_dim_size, spatial_dim in zip(block_size, spatial_dims):
        final_shape[spatial_dim] //= block_dim_size
        final_shape[channel_dim] *= block_dim_size

    y = x_permuted.reshape(final_shape)

    return y


def cat_unpool(x, channel_dim, spatial_dims, block_size):
    """
    Expand spatial dimensions by unfolding channel values back into spatial dimensions.

    This is the exact inverse of cat_pool: values that were folded into the channel
    dimension are redistributed back into the spatial dimensions.

    Args:
        x (torch.Tensor): Input tensor of any number of dimensions.
        channel_dim (int): Index of the channel dimension. Supports negative indexing.
        spatial_dims (list[int]): Indices of the spatial dimensions to unpool over.
            Supports negative indexing. Must not overlap with channel_dim.
        block_size (list[int]): block size for each spatial dimension. The channel
            dimension size must be divisible by the product of all block sizes.

    Returns:
        torch.Tensor: Output tensor with the same number of dimensions as x, where
            each spatial dimension is multiplied by its block size, and the channel
            dimension is divided by the product of all block sizes.

    Raises:
        ValueError: If channel_dim and spatial_dims overlap.
        ValueError: If block_size and spatial_dims have different lengths.
        ValueError: If the channel dimension is not divisible by the product of all
            block sizes.

    Example:
        >>> x = torch.randn(10, 12, 2, 2)
        >>> y = cat_unpool(x, channel_dim=1, spatial_dims=[2, 3], block_size=[2, 2])
        >>> y.shape
        torch.Size([10, 3, 4, 4])
    """
    # Validation
    dim_usage = [0] * x.dim()
    dim_usage[channel_dim] += 1
    for dim in spatial_dims:
        dim_usage[dim] += 1
    if max(dim_usage) > 1:
        raise ValueError(f"Redundant dimension in [{channel_dim}], {spatial_dims} for torch of size {x.size()}.")

    if len(block_size) != len(spatial_dims):
        raise ValueError(f"block size and spatial dims do not match ({len(block_size)} != {len(spatial_dims)}).")

    total_block_size = 1
    for block_dim_size in block_size:
        total_block_size *= block_dim_size

    if x.size(channel_dim) % total_block_size != 0:
        raise ValueError(f"Channel size ({x.size(channel_dim)}) must be divisible by block size ({block_size}).")

    # 1. Split channel dim into [block_dims..., reduced_channel]
    split_dims = [[dim_size] for dim_size in x.size()]
    split_dims[channel_dim] = [*block_size, x.size(channel_dim) // total_block_size]

    # 2. Reshape: introduce block dimensions as explicit dimensions
    x_split = x.reshape([dim_size for dim_sizes in split_dims for dim_size in dim_sizes])

    # 3. Build mapping: original dim -> indices in x_split
    i = 0
    orig_to_split_indices = []
    for dim_sizes in split_dims:
        orig_to_split_indices.append(list(range(i, i + len(dim_sizes))))
        i += len(dim_sizes)

    # 4. Move block indices back to their spatial dims
    block_indices = orig_to_split_indices[channel_dim][:-1]  # all but the last (reduced channel)
    orig_to_split_indices[channel_dim] = [orig_to_split_indices[channel_dim][-1]]
    for block_idx, spatial_dim in zip(block_indices, spatial_dims):
        orig_to_split_indices[spatial_dim].append(block_idx)

    # 5. Permute to move block indices back to spatial dims
    perm = [idx for dim_indices in orig_to_split_indices for idx in dim_indices]
    x_permuted = x_split.permute(perm)

    # 6. Final reshape: merge block dimensions back into spatial dims
    final_shape = list(x.size())
    for block_dim_size, spatial_dim in zip(block_size, spatial_dims):
        final_shape[spatial_dim] *= block_dim_size
        final_shape[channel_dim] //= block_dim_size

    y = x_permuted.reshape(final_shape)

    return y
