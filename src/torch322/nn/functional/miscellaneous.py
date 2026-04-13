import torch


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
