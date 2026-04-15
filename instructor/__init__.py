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
from .processing.function_calls import OpenAISchema, openai_schema
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

# Backward compatibility: Re-export removed functions
from .processing.response import handle_response_model
from .dsl.parallel import handle_parallel_model

__all__ = [
    "Instructor",
    "Image",
    "Audio",
    "from_openai",
    "from_litellm",
    "from_provider",
    "AsyncInstructor",
    "Provider",
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
    "client",  # Backward compatibility
    # Backward compatibility exports
    "handle_response_model",
    "handle_parallel_model",
]

# Backward compatibility: Make instructor.client available as an attribute
# This allows code like `instructor.client.Instructor` to work
from . import client


# Provider factory functions are loaded lazily via __getattr__ to avoid pulling
# in heavy SDK dependencies (anthropic, boto3, google-genai, etc.) at import time.
# See https://github.com/567-labs/instructor/issues/2205

_LAZY_PROVIDERS: dict[str, tuple[str, str | list[str]]] = {
    # name -> (module_path, required_spec_or_specs)
    "from_anthropic": (".providers.anthropic.client", "anthropic"),
    "from_gemini": (".providers.gemini.client", ["google", "google.generativeai"]),
    "from_fireworks": (".providers.fireworks.client", "fireworks"),
    "from_cerebras": (".providers.cerebras.client", "cerebras"),
    "from_groq": (".providers.groq.client", "groq"),
    "from_mistral": (".providers.mistral.client", "mistralai"),
    "from_cohere": (".providers.cohere.client", "cohere"),
    "from_vertexai": (".providers.vertexai.client", ["vertexai", "jsonref"]),
    "from_bedrock": (".providers.bedrock.client", "boto3"),
    "from_writer": (".providers.writer.client", "writerai"),
    "from_xai": (".providers.xai.client", "xai_sdk"),
    "from_perplexity": (".providers.perplexity.client", "openai"),
    "from_genai": (".providers.genai.client", ["google", "google.genai"]),
}

# Populate __all__ based on available specs without importing the providers
for _name, (_, _specs) in _LAZY_PROVIDERS.items():
    _spec_list = [_specs] if isinstance(_specs, str) else _specs
    if all(importlib.util.find_spec(s) is not None for s in _spec_list):
        __all__ += [_name]


def __getattr__(name: str):
    if name in _LAZY_PROVIDERS:
        module_path, specs = _LAZY_PROVIDERS[name]
        spec_list = [specs] if isinstance(specs, str) else specs
        try:
            specs_found = all(
                importlib.util.find_spec(s) is not None for s in spec_list
            )
        except (ValueError, ModuleNotFoundError):
            specs_found = False
        if not specs_found:
            raise AttributeError(
                f"module 'instructor' has no attribute {name!r} "
                f"(missing optional dependency)"
            )
        try:
            mod = importlib.import_module(module_path, package=__name__)
            attr = getattr(mod, name)
        except Exception as exc:
            raise AttributeError(
                f"module 'instructor' has no attribute {name!r}"
            ) from exc
        # Cache on the module so __getattr__ is not called again
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'instructor' has no attribute {name!r}")
