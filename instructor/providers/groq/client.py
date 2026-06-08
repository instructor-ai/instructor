"""Compatibility facade for ``instructor.providers.groq.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_groq"]
__getattr__ = make_getattr("groq", ("client",))
