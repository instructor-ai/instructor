"""Compatibility facade for ``instructor.providers.gemini.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_gemini"]
__getattr__ = make_getattr("gemini", ("client",))
