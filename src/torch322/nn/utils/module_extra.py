from torch.nn import Module


def call_module_method(
        module: Module,
        method_name,
        required=True,
        args=None,
        kwargs=None,
):
    args = args or []
    kwargs = kwargs or {}
    method = getattr(module, method_name, None)
    if callable(method):
        return method(*args, **kwargs)
    else:
        if required:
            raise RuntimeError(f"Method {method_name} not found in module {module}")
        else:
            return None


def call_module_method_recursive(
        module: Module,
        method_name,
        required=True,
        args=None,
        kwargs=None,
):
    def to_apply(inner_module):
        return call_module_method(inner_module, method_name, required=required, args=args, kwargs=kwargs)

    module.apply(to_apply)
