"""Compatibility facade for ``instructor.providers.cohere.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_cohere"]
__getattr__ = make_getattr("cohere", ("client",))
