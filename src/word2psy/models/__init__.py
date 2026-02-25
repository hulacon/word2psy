"""Lazy model registry — imports only happen when a model class is accessed."""


def __getattr__(name):
    _registry = {
        "CLIPTextModel": ".clip_text",
        "LexicalNormsModel": ".lexical_norms",
    }
    if name in _registry:
        import importlib

        module = importlib.import_module(_registry[name], __package__)
        return getattr(module, name)
    if name == "__all__":
        return list(_registry.keys())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
