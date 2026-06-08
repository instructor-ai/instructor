"""Compatibility facade for ``instructor.providers.fireworks.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_fireworks"]
__getattr__ = make_getattr("fireworks", ("client",))
