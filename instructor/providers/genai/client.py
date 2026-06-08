"""Compatibility facade for ``instructor.providers.genai.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_genai"]
__getattr__ = make_getattr("genai", ("client",))
