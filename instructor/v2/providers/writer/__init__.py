"""v2 Writer provider."""

try:
    from instructor.v2.providers.writer.client import from_writer
except ImportError:
    from_writer = None  # type: ignore

__all__ = ["from_writer"]
