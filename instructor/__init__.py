import importlib.util

__version__ = "1.15.1"

from .mode import Mode
from .processing.multimodal import Image, Audio

from .dsl import (
    CitationMixin,
    Maybe,
    Partial,
    IterableModel,
)

from .validation import llm_validator, openai_moderation
from .processing.function_calls import (
    ResponseSchema,
    response_schema,
    OpenAISchema,
    openai_schema,
)
from .processing.schema import (
    generate_openai_schema,
    generate_anthropic_schema,
    generate_gemini_schema,
)
from .core.patch import apatch, patch
from .core.client import (
    Instructor,
    AsyncInstructor,
    from_openai,
    from_litellm,
)
from .core import hooks
from .utils.providers import Provider
from .auto_client import from_provider
from .batch import BatchProcessor, BatchRequest, BatchJob
from .distil import FinetuneFormat, Instructions

__all__ = [
    "Instructor",
    "Image",
    "Audio",
    "from_openai",
    "from_litellm",
    "from_provider",
    "AsyncInstructor",
    "Provider",
    "ResponseSchema",
    "response_schema",
    "OpenAISchema",
    "CitationMixin",
    "IterableModel",
    "Maybe",
    "Partial",
    "openai_schema",
    "generate_openai_schema",
    "generate_anthropic_schema",
    "generate_gemini_schema",
    "Mode",
    "patch",
    "apatch",
    "FinetuneFormat",
    "Instructions",
    "BatchProcessor",
    "BatchRequest",
    "BatchJob",
    "llm_validator",
    "openai_moderation",
    "hooks",
    "client",
]

from . import client


if importlib.util.find_spec("anthropic") is not None:
    from .v2.providers.anthropic.client import from_anthropic

    __all__ += ["from_anthropic"]

# Keep from_gemini for backward compatibility but it's deprecated
if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.generativeai") is not None
):
    from .v2.providers.gemini.client import from_gemini

    __all__ += ["from_gemini"]

if importlib.util.find_spec("fireworks") is not None:
    from .v2.providers.fireworks.client import from_fireworks

    __all__ += ["from_fireworks"]

if importlib.util.find_spec("cerebras") is not None:
    from .v2.providers.cerebras.client import from_cerebras

    __all__ += ["from_cerebras"]

if importlib.util.find_spec("groq") is not None:
    from .v2.providers.groq.client import from_groq

    __all__ += ["from_groq"]

if importlib.util.find_spec("mistralai") is not None:
    from .v2.providers.mistral.client import from_mistral

    __all__ += ["from_mistral"]

if importlib.util.find_spec("cohere") is not None:
    from .v2.providers.cohere.client import from_cohere

    __all__ += ["from_cohere"]

def from_vertexai(*args, **kwargs):
    from .v2.providers.vertexai.client import from_vertexai as _from_vertexai

    return _from_vertexai(*args, **kwargs)


__all__ += ["from_vertexai"]

if importlib.util.find_spec("boto3") is not None:
    from .v2.providers.bedrock.client import from_bedrock

    __all__ += ["from_bedrock"]

if importlib.util.find_spec("writerai") is not None:
    from .v2.providers.writer.client import from_writer

    __all__ += ["from_writer"]

if importlib.util.find_spec("xai_sdk") is not None:
    from .v2.providers.xai.client import from_xai

    __all__ += ["from_xai"]

if importlib.util.find_spec("openai") is not None:
    from .v2.providers.perplexity.client import from_perplexity

    __all__ += ["from_perplexity"]


if importlib.util.find_spec("google") and importlib.util.find_spec("google.genai") is not None:
    from .v2.providers.genai.client import from_genai

    __all__ += ["from_genai"]
