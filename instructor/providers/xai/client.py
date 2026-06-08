"""Compatibility facade for ``instructor.providers.xai.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_xai"]
__getattr__ = make_getattr("xai", ("client",))
