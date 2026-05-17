from __future__ import annotations

import importlib.util
from importlib import import_module
from typing import Any

__version__ = "1.15.2"

__all__ = [
    "Instructor",
    "Image",
    "Audio",
    "from_openai",
    "from_anyscale",
    "from_together",
    "from_databricks",
    "from_deepseek",
    "from_openrouter",
    "from_litellm",
    "from_vertexai",
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
    "v2",
]

_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "Instructor": (".core.client", "Instructor"),
    "AsyncInstructor": (".core.client", "AsyncInstructor"),
    "from_openai": (".v2.providers.openai.client", "from_openai"),
    "from_anyscale": (".v2.providers.openai.client", "from_anyscale"),
    "from_together": (".v2.providers.openai.client", "from_together"),
    "from_databricks": (".v2.providers.openai.client", "from_databricks"),
    "from_deepseek": (".v2.providers.openai.client", "from_deepseek"),
    "from_openrouter": (".v2.providers.openrouter.client", "from_openrouter"),
    "from_litellm": (".v2.providers.litellm.client", "from_litellm"),
    "Mode": (".mode", "Mode"),
    "patch": (".core.patch", "patch"),
    "apatch": (".core.patch", "apatch"),
    "hooks": (".core.hooks", None),
    "v2": (".v2", None),
    "Image": (".v2.core.multimodal", "Image"),
    "Audio": (".v2.core.multimodal", "Audio"),
    "CitationMixin": (".v2.dsl", "CitationMixin"),
    "IterableModel": (".v2.dsl", "IterableModel"),
    "Maybe": (".v2.dsl", "Maybe"),
    "Partial": (".v2.dsl", "Partial"),
    "ResponseSchema": (".v2.core.function_calls", "ResponseSchema"),
    "response_schema": (".v2.core.function_calls", "response_schema"),
    "OpenAISchema": (".v2.core.function_calls", "OpenAISchema"),
    "openai_schema": (".v2.core.function_calls", "openai_schema"),
    "generate_openai_schema": (".v2.core.schema", "generate_openai_schema"),
    "generate_anthropic_schema": (".v2.core.schema", "generate_anthropic_schema"),
    "generate_gemini_schema": (".v2.core.schema", "generate_gemini_schema"),
    "llm_validator": (".v2.validation", "llm_validator"),
    "openai_moderation": (".v2.validation", "openai_moderation"),
    "Provider": (".utils.providers", "Provider"),
    "from_provider": (".auto_client", "from_provider"),
    "BatchProcessor": (".batch", "BatchProcessor"),
    "BatchRequest": (".batch", "BatchRequest"),
    "BatchJob": (".batch", "BatchJob"),
    "FinetuneFormat": (".distil", "FinetuneFormat"),
    "Instructions": (".distil", "Instructions"),
    "from_anthropic": (".v2.providers.anthropic.client", "from_anthropic"),
    "from_gemini": (".v2.providers.gemini.client", "from_gemini"),
    "from_fireworks": (".v2.providers.fireworks.client", "from_fireworks"),
    "from_cerebras": (".v2.providers.cerebras.client", "from_cerebras"),
    "from_groq": (".v2.providers.groq.client", "from_groq"),
    "from_mistral": (".v2.providers.mistral.client", "from_mistral"),
    "from_cohere": (".v2.providers.cohere.client", "from_cohere"),
    "from_vertexai": (".v2.providers.vertexai.client", "from_vertexai"),
    "from_bedrock": (".v2.providers.bedrock.client", "from_bedrock"),
    "from_writer": (".v2.providers.writer.client", "from_writer"),
    "from_xai": (".v2.providers.xai.client", "from_xai"),
    "from_perplexity": (".v2.providers.perplexity.client", "from_perplexity"),
    "from_genai": (".v2.providers.genai.client", "from_genai"),
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_path, package=__name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def _add_optional_export(name: str, *packages: str) -> None:
    if all(importlib.util.find_spec(package) is not None for package in packages):
        __all__.append(name)


_add_optional_export("from_anthropic", "anthropic")
_add_optional_export("from_gemini", "google", "google.generativeai")
_add_optional_export("from_fireworks", "fireworks")
_add_optional_export("from_cerebras", "cerebras")
_add_optional_export("from_groq", "groq")
_add_optional_export("from_mistral", "mistralai")
_add_optional_export("from_cohere", "cohere")
_add_optional_export("from_bedrock", "boto3")
_add_optional_export("from_writer", "writerai")
_add_optional_export("from_xai", "xai_sdk")
_add_optional_export("from_perplexity", "openai")
_add_optional_export("from_genai", "google", "google.genai")
