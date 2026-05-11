"""v2 Anthropic provider."""

try:
    from instructor.v2.providers.anthropic.client import from_anthropic
except ImportError:
    from_anthropic = None  # type: ignore

__all__ = ["from_anthropic"]
