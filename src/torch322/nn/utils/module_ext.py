from torch.nn import Module


def call_method(module: Module, method_name, required, *args, **kwargs):
    method = getattr(module, method_name, None)
    if callable(method):
        return method(*args, **kwargs)
    else:
        if required:
            raise RuntimeError(f"Method {method_name} not found")
        else:
            return None


def call_method_recursive(module: Module, method_name, required, *args, **kwargs):

    def to_apply(sub_module):
        return call_method(sub_module, method_name, required, *args, **kwargs)

    module.apply(to_apply)
