"""Small explicit registry; additional backends register a zero-argument factory."""
from .dense import DenseAnalytic

_FACTORIES = {"dense_analytic": DenseAnalytic}


def register_backend(name, factory):
    if not name or name in _FACTORIES:
        raise ValueError(f"Empty or duplicate backend name: {name}")
    _FACTORIES[name] = factory


def create_backend(name):
    if name not in _FACTORIES:
        raise ValueError(f"Unknown hardware backend {name!r}; choices: {sorted(_FACTORIES)}")
    return _FACTORIES[name]()
