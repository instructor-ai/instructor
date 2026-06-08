"""Compatibility facade for ``instructor.providers.cerebras.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_cerebras"]
__getattr__ = make_getattr("cerebras", ("client",))
