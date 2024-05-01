import importlib.util

from .client import (
    AsyncInstructor,
    Instructor,
    Provider,
    from_litellm,
    from_openai,
)
from .distil import FinetuneFormat, Instructions
from .dsl import (
    CitationMixin,
    IterableModel,
    Maybe,
    Partial,
    llm_validator,
    openai_moderation,
)
from .function_calls import OpenAISchema, openai_schema
from .mode import Mode
from .patch import apatch, patch
from .process_response import handle_parallel_model, handle_response_model

__all__ = [
    "Instructor",
    "from_openai",
    "from_litellm",
    "AsyncInstructor",
    "Provider",
    "OpenAISchema",
    "CitationMixin",
    "IterableModel",
    "Maybe",
    "Partial",
    "openai_schema",
    "Mode",
    "patch",
    "apatch",
    "llm_validator",
    "openai_moderation",
    "FinetuneFormat",
    "Instructions",
    "handle_parallel_model",
    "handle_response_model",
]


if importlib.util.find_spec("anthropic") is not None:
    from .client_anthropic import from_anthropic

    __all__ += ["from_anthropic"]

if importlib.util.find_spec("groq") is not None:
    from .client_groq import from_groq

    __all__ += ["from_groq"]

if importlib.util.find_spec("mistralai") is not None:
    from .client_mistral import from_mistral

    __all__ += ["from_mistral"]

if importlib.util.find_spec("cohere") is not None:
    from .client_cohere import from_cohere

    __all__ += ["from_cohere"]
