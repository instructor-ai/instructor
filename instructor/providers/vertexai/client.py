"""Compatibility facade for ``instructor.providers.vertexai.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_vertexai"]
__getattr__ = make_getattr("vertexai", ("client",))
