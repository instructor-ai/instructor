from __future__ import annotations

import importlib.util
from importlib import import_module
from typing import Any

__version__ = "1.15.1"

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

# Provider availability checks (lightweight, no SDK imports)
if importlib.util.find_spec("anthropic") is not None:
    __all__ += ["from_anthropic"]

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.generativeai") is not None
):
    __all__ += ["from_gemini"]

if importlib.util.find_spec("fireworks") is not None:
    __all__ += ["from_fireworks"]

if importlib.util.find_spec("cerebras") is not None:
    __all__ += ["from_cerebras"]

if importlib.util.find_spec("groq") is not None:
    __all__ += ["from_groq"]

if importlib.util.find_spec("mistralai") is not None:
    __all__ += ["from_mistral"]

if importlib.util.find_spec("cohere") is not None:
    __all__ += ["from_cohere"]

if all(importlib.util.find_spec(pkg) for pkg in ("vertexai", "jsonref")):
    __all__ += ["from_vertexai"]

if importlib.util.find_spec("boto3") is not None:
    __all__ += ["from_bedrock"]

if importlib.util.find_spec("writerai") is not None:
    __all__ += ["from_writer"]

if importlib.util.find_spec("xai_sdk") is not None:
    __all__ += ["from_xai"]

if importlib.util.find_spec("openai") is not None:
    __all__ += ["from_perplexity"]

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.genai") is not None
):
    __all__ += ["from_genai"]


_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    # Core
    "Instructor": (".core.client", "Instructor"),
    "AsyncInstructor": (".core.client", "AsyncInstructor"),
    "from_openai": (".core.client", "from_openai"),
    "from_litellm": (".core.client", "from_litellm"),
    "Mode": (".mode", "Mode"),
    "patch": (".core.patch", "patch"),
    "apatch": (".core.patch", "apatch"),
    "hooks": (".core.hooks", None),
    # Multimodal
    "Image": (".processing.multimodal", "Image"),
    "Audio": (".processing.multimodal", "Audio"),
    # DSL
    "CitationMixin": (".dsl", "CitationMixin"),
    "IterableModel": (".dsl", "IterableModel"),
    "Maybe": (".dsl", "Maybe"),
    "Partial": (".dsl", "Partial"),
    # Processing
    "OpenAISchema": (".processing.function_calls", "OpenAISchema"),
    "openai_schema": (".processing.function_calls", "openai_schema"),
    "generate_openai_schema": (".processing.schema", "generate_openai_schema"),
    "generate_anthropic_schema": (".processing.schema", "generate_anthropic_schema"),
    "generate_gemini_schema": (".processing.schema", "generate_gemini_schema"),
    # Validation
    "llm_validator": (".validation", "llm_validator"),
    "openai_moderation": (".validation", "openai_moderation"),
    # Utilities
    "Provider": (".utils.providers", "Provider"),
    "from_provider": (".auto_client", "from_provider"),
    # Batch
    "BatchProcessor": (".batch", "BatchProcessor"),
    "BatchRequest": (".batch", "BatchRequest"),
    "BatchJob": (".batch", "BatchJob"),
    # Distil
    "FinetuneFormat": (".distil", "FinetuneFormat"),
    "Instructions": (".distil", "Instructions"),
    # Backward compatibility
    "client": (".client", None),
    "handle_response_model": (".processing.response", "handle_response_model"),
    "handle_parallel_model": (".dsl.parallel", "handle_parallel_model"),
    # Provider factories
    "from_anthropic": (".providers.anthropic.client", "from_anthropic"),
    "from_gemini": (".providers.gemini.client", "from_gemini"),
    "from_fireworks": (".providers.fireworks.client", "from_fireworks"),
    "from_cerebras": (".providers.cerebras.client", "from_cerebras"),
    "from_groq": (".providers.groq.client", "from_groq"),
    "from_mistral": (".providers.mistral.client", "from_mistral"),
    "from_cohere": (".providers.cohere.client", "from_cohere"),
    "from_vertexai": (".providers.vertexai.client", "from_vertexai"),
    "from_bedrock": (".providers.bedrock.client", "from_bedrock"),
    "from_writer": (".providers.writer.client", "from_writer"),
    "from_xai": (".providers.xai.client", "from_xai"),
    "from_perplexity": (".providers.perplexity.client", "from_perplexity"),
    "from_genai": (".providers.genai.client", "from_genai"),
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_path, package=__name__)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
