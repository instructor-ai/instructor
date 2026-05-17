"""v2 Groq provider."""

try:
    from instructor.v2.providers.groq.client import from_groq
except ImportError:
    from_groq = None  # type: ignore

__all__ = ["from_groq"]
