

class TensorSizeError(Exception):
    pass


def check_tensor_sizes(tensors, sizes, raise_exception=False, values=None):
    if len(tensors) != len(sizes):
        exception = ValueError(f"Number of tensors ({len(tensors)}) and sizes ({len(sizes)}) are different.")
        if raise_exception:
            raise exception
        return False, exception
    if values is None:
        values = {}
    for tensor, size in zip(tensors, sizes):
        current_size = tensor.size()
        if len(current_size) != len(size):
            exception = TensorSizeError(f"Number of dimensions in tensor ({len(current_size)}) and expected size "
                                        f"({len(size)}) are different.")
            if raise_exception:
                raise exception
            return False, exception
        for current_dimension, ref_dimension in zip(current_size, size):
            if type(ref_dimension) == str:
                if ref_dimension in values:
                    ref_dimension = values[ref_dimension]
                else:
                    values[ref_dimension] = current_dimension
                    continue
            elif ref_dimension is None or ref_dimension < 0:
                continue
            if ref_dimension != current_dimension:
                exception = TensorSizeError(f"Tensor size ({current_size}) not compatible with expected size "
                                            f"({size}). (values:{values})")
                if raise_exception:
                    raise exception
                return False, exception

    return True, values


class TensorSizeChecker:
    def __init__(self, raise_exception=False, values=None):
        self.raise_exception = raise_exception
        self.values = {} if values is None else values

    def check(self, tensors, sizes, raise_exception=None):
        if raise_exception is None:
            raise_exception = self.raise_exception
        check_tensor_sizes(tensors, sizes, raise_exception=raise_exception, values=self.values)
