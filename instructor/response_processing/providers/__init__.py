"""Provider-specific response handlers."""

from .anthropic import register_anthropic_handlers
from .bedrock import register_bedrock_handlers
from .cerebras import register_cerebras_handlers
from .cohere import register_cohere_handlers
from .fireworks import register_fireworks_handlers
from .gemini import register_gemini_handlers
from .genai import register_genai_handlers
from .mistral import register_mistral_handlers
from .openai import register_openai_handlers
from .vertexai import register_vertexai_handlers
from .writer import register_writer_handlers
from .perplexity import register_perplexity_handlers

__all__ = [
    "register_anthropic_handlers",
    "register_bedrock_handlers",
    "register_cerebras_handlers",
    "register_cohere_handlers",
    "register_fireworks_handlers",
    "register_gemini_handlers",
    "register_genai_handlers",
    "register_mistral_handlers",
    "register_openai_handlers",
    "register_vertexai_handlers",
    "register_writer_handlers",
    "register_perplexity_handlers",
]


def register_all_handlers() -> None:
    """Register all provider handlers."""
    register_openai_handlers()
    register_anthropic_handlers()
    register_bedrock_handlers()
    register_cerebras_handlers()
    register_cohere_handlers()
    register_fireworks_handlers()
    register_gemini_handlers()
    register_genai_handlers()
    register_mistral_handlers()
    register_vertexai_handlers()
    register_writer_handlers()
    register_perplexity_handlers()
