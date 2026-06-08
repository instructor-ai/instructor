"""Compatibility facade for ``instructor.providers.anthropic.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_anthropic"]
__getattr__ = make_getattr("anthropic", ("client",))
