"""v2 OpenAI provider."""

try:
    from instructor.v2.providers.openai.client import from_openai
except ImportError:
    from_openai = None  # type: ignore

__all__ = ["from_openai"]
