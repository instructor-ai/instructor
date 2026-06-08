"""Compatibility facade for ``instructor.providers.mistral.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_mistral"]
__getattr__ = make_getattr("mistral", ("client",))
