"""Compatibility facade for ``instructor.providers.perplexity.client``."""

from instructor.providers._compat import make_getattr

__all__ = ["from_perplexity"]
__getattr__ = make_getattr("perplexity", ("client",))
