"""Opt-in exact-code observer. Disabled path has no tensor work or torch import."""

from contextlib import contextmanager
from contextvars import ContextVar

_OBSERVER = ContextVar("hardware_quant_observer", default=None)


@contextmanager
def observe_quantized_operands(callback):
    token = _OBSERVER.set(callback)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def emit_quantized_operands(
    layer, a, b, a_code, b_code, output, output_code, *, linear
):
    observer = _OBSERVER.get()
    if observer is not None:
        observer(layer, a, b, a_code, b_code, output, output_code, linear=linear)
