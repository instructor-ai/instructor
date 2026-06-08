"""Compatibility facade for ``instructor.providers.bedrock.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_bedrock"]
__getattr__ = make_getattr("bedrock", ("client",))
