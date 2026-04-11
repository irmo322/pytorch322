import torch


class TensorSizeError(Exception):
    pass


def check_tensor_size(tensor_or_size_s, expected_size_s, values=None):
    """
    Check given torch tensor size against expected size which can be a mix of integer values and string keys.

    When compared to a positive integer value, the size of a dimension must be equal to that value.
    When compared to None or a strictly negative value, the size of a dimension does not matter.
    When compared to a key, the size of a dimension must be equal to every other size of dimension compared to that same
    key.
    Instead of one tensor and one size (as a list-like object), this function also accept nested collections (list or
    dict) of tensors and sizes.
    This function returns a dict mapping string keys found in expected sizes to integer values from tensor sizes if
    the size of the tensor matchs the expected size, or raise a TensorSizeError otherwise.
    User can optionally provide a prefilled values map.

    :param tensor_or_size_s: torch.Tensor, torch.Size or (possibly nested) collection of them.
    :param expected_size_s: list-like object containing integers and/or strings, or collection of such size objects.
    :param values: optional. dict mapping strings to integers.
    :return: the dict mapping string keys found in expected sizes to integer values from tensor sizes. Initialized from
        values if provided.
    """
    if values is None:
        values = {}

    if isinstance(tensor_or_size_s, torch.Tensor):
        tensor_or_size_s = tensor_or_size_s.size()

    if isinstance(tensor_or_size_s, torch.Size):
        if len(tensor_or_size_s) != len(expected_size_s):
            raise TensorSizeError(
                f"Number of dimensions in tensor ({len(tensor_or_size_s)}) and expected size ({len(expected_size_s)})"
                f"are different.")
        for tensor_dimension_size, ref_dimension_size in zip(tensor_or_size_s, expected_size_s):
            if type(ref_dimension_size) == str:
                if ref_dimension_size in values:
                    ref_dimension_size = values[ref_dimension_size]
                else:
                    values[ref_dimension_size] = tensor_dimension_size
                    continue
            elif ref_dimension_size is None or ref_dimension_size < 0:
                continue
            if ref_dimension_size != tensor_dimension_size:
                raise TensorSizeError(
                    f"Tensor size ({tensor_or_size_s}) not compatible with expected size ({expected_size_s}). "
                    f"(values:{values})")

    else:
        # expect tensor_or_size_s and expected_size_s to be collections

        if len(tensor_or_size_s) != len(expected_size_s):
            raise TensorSizeError(
                f"Collections of different lengths ({len(tensor_or_size_s)} != {len(expected_size_s)}).")

        keys = tensor_or_size_s.keys() if type(tensor_or_size_s) is dict else range(len(tensor_or_size_s))
        for key in keys:
            check_tensor_size(tensor_or_size_s[key], expected_size_s[key], values=values)

    return values


class TensorSizeChecker:
    def __init__(self, values=None):
        self.values = {} if values is None else values

    def check(self, tensor_s, size_s):
        check_tensor_size(tensor_s, size_s, values=self.values)
