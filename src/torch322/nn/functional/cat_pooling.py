
def cat_pool(x, channel_dim, spatial_dims, kernel_size):
    # Validation
    dim_usage = [0] * x.dim()
    dim_usage[channel_dim] += 1
    for dim in spatial_dims:
        dim_usage[dim] += 1
    if max(dim_usage) > 1:
        raise ValueError(f"Redundant dimension in [{channel_dim}], {spatial_dims} for torch of size {x.size()}.")

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
