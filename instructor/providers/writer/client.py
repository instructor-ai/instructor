"""Compatibility facade for ``instructor.providers.writer.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_writer"]
__getattr__ = make_getattr("writer", ("client",))
