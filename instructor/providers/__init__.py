"""Provider implementations for instructor."""

from __future__ import annotations

import importlib.util
from importlib import import_module
from typing import Any

__all__: list[str] = []

_LAZY_PROVIDER_EXPORTS: dict[str, tuple[str, str]] = {
    "from_anthropic": (".anthropic.client", "from_anthropic"),
    "from_bedrock": (".bedrock.client", "from_bedrock"),
    "from_cerebras": (".cerebras.client", "from_cerebras"),
    "from_cohere": (".cohere.client", "from_cohere"),
    "from_fireworks": (".fireworks.client", "from_fireworks"),
    "from_gemini": (".gemini.client", "from_gemini"),
    "from_genai": (".genai.client", "from_genai"),
    "from_groq": (".groq.client", "from_groq"),
    "from_mistral": (".mistral.client", "from_mistral"),
    "from_perplexity": (".perplexity.client", "from_perplexity"),
    "from_vertexai": (".vertexai.client", "from_vertexai"),
    "from_writer": (".writer.client", "from_writer"),
    "from_xai": (".xai.client", "from_xai"),
}

# Conditional exports based on installed packages
if importlib.util.find_spec("anthropic") is not None:
    __all__.append("from_anthropic")

if importlib.util.find_spec("boto3") is not None:
    __all__.append("from_bedrock")

if importlib.util.find_spec("cerebras") is not None:
    __all__.append("from_cerebras")

if importlib.util.find_spec("cohere") is not None:
    __all__.append("from_cohere")

if importlib.util.find_spec("fireworks") is not None:
    __all__.append("from_fireworks")

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.generativeai") is not None
):
    __all__.append("from_gemini")

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.genai") is not None
):
    __all__.append("from_genai")

if importlib.util.find_spec("groq") is not None:
    __all__.append("from_groq")

if importlib.util.find_spec("mistralai") is not None:
    __all__.append("from_mistral")

if importlib.util.find_spec("openai") is not None:
    __all__.append("from_perplexity")

if all(importlib.util.find_spec(pkg) for pkg in ("vertexai", "jsonref")):
    __all__.append("from_vertexai")

if importlib.util.find_spec("writerai") is not None:
    __all__.append("from_writer")

if importlib.util.find_spec("xai_sdk") is not None:
    __all__.append("from_xai")


def __getattr__(name: str) -> Any:
    if name in _LAZY_PROVIDER_EXPORTS and name in __all__:
        module_path, attr_name = _LAZY_PROVIDER_EXPORTS[name]
        try:
            module = import_module(module_path, package=__name__)
        except Exception as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
