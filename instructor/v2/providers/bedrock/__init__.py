"""v2 Bedrock provider."""

try:
    from instructor.v2.providers.bedrock.client import from_bedrock
except ImportError:
    from_bedrock = None  # type: ignore

__all__ = ["from_bedrock"]
