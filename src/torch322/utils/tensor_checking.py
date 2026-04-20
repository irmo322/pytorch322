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
    When compared to a key starting with '*', the key captures zero or more consecutive dimensions as a tuple.
    This tuple must be equal to every other tuple captured by the same key across different tensors.
    There can be at most one '*' key in a given expected size.
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

        star_count = 0
        for ref_dimension_size in expected_size_s:
            star_count += type(ref_dimension_size) == str and ref_dimension_size.startswith("*")

        if star_count > 1:
            raise ValueError(f"Multiple keys starting with '*' found in expected size ({expected_size_s}), "
                             f"expect no more than one.")

        if star_count == 0 and len(tensor_or_size_s) != len(expected_size_s):
            raise TensorSizeError(
                f"Number of dimensions in tensor ({len(tensor_or_size_s)}) and expected size ({len(expected_size_s)}) "
                f"are different.")
        if star_count == 1 and len(tensor_or_size_s) < len(expected_size_s) - 1:
            raise TensorSizeError(
                f"Number of dimensions in tensor ({len(tensor_or_size_s)}) incompatible with expected size "
                f"{expected_size_s}.")

        before_star = True
        for i, ref_dimension_size in enumerate(expected_size_s):
            if ref_dimension_size is None or (type(ref_dimension_size) != str and ref_dimension_size < 0):
                continue
            if type(ref_dimension_size) == str and ref_dimension_size.startswith("*"):
                tensor_dimension_size = tuple(
                    tensor_or_size_s[i: i + len(tensor_or_size_s) - (len(expected_size_s) - 1)])
                before_star = False
            else:
                tensor_dimension_size = (
                    tensor_or_size_s[i] if before_star
                    else tensor_or_size_s[i - len(expected_size_s)])
            if type(ref_dimension_size) == str:
                if ref_dimension_size in values:
                    ref_dimension_size = values[ref_dimension_size]
                else:
                    values[ref_dimension_size] = tensor_dimension_size
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
            try:
                expected_size = expected_size_s[key]
            except KeyError:
                raise TensorSizeError(f"key ({key}) not found in expected size ({expected_size_s})")
            check_tensor_size(tensor_or_size_s[key], expected_size, values=values)

    return values


class TensorSizeChecker:
    def __init__(self, values=None):
        self.values = {} if values is None else values

    def check(self, tensor_s, size_s):
        check_tensor_size(tensor_s, size_s, values=self.values)
