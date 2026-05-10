"""Instructor v2 public exports with lazy loading."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Mode",
    "Provider",
    "ModeHandler",
    "ModeHandlers",
    "ModeRegistry",
    "mode_registry",
    "normalize_mode",
    "patch_v2",
    "register_mode_handler",
    "ReaskHandler",
    "RequestHandler",
    "ResponseParser",
    "providers",
    "from_anthropic",
    "from_anyscale",
    "from_bedrock",
    "from_cerebras",
    "from_cohere",
    "from_databricks",
    "from_deepseek",
    "from_fireworks",
    "from_gemini",
    "from_genai",
    "from_groq",
    "from_mistral",
    "from_openai",
    "from_openrouter",
    "from_perplexity",
    "from_together",
    "from_vertexai",
    "from_writer",
    "from_xai",
]

_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    "Mode": ("instructor.v2.core.mode", "Mode"),
    "Provider": ("instructor.v2.core.providers", "Provider"),
    "ModeHandler": ("instructor.v2.core.handler", "ModeHandler"),
    "ModeHandlers": ("instructor.v2.core.registry", "ModeHandlers"),
    "ModeRegistry": ("instructor.v2.core.registry", "ModeRegistry"),
    "mode_registry": ("instructor.v2.core.registry", "mode_registry"),
    "normalize_mode": ("instructor.v2.core.registry", "normalize_mode"),
    "patch_v2": ("instructor.v2.core.patch", "patch_v2"),
    "register_mode_handler": (
        "instructor.v2.core.decorators",
        "register_mode_handler",
    ),
    "ReaskHandler": ("instructor.v2.core.protocols", "ReaskHandler"),
    "RequestHandler": ("instructor.v2.core.protocols", "RequestHandler"),
    "ResponseParser": ("instructor.v2.core.protocols", "ResponseParser"),
    "providers": ("instructor.v2.providers", None),
    "from_anthropic": (
        "instructor.v2.providers.anthropic.client",
        "from_anthropic",
    ),
    "from_anyscale": ("instructor.v2.providers.openai.client", "from_anyscale"),
    "from_bedrock": ("instructor.v2.providers.bedrock.client", "from_bedrock"),
    "from_cerebras": (
        "instructor.v2.providers.cerebras.client",
        "from_cerebras",
    ),
    "from_cohere": ("instructor.v2.providers.cohere.client", "from_cohere"),
    "from_databricks": (
        "instructor.v2.providers.openai.client",
        "from_databricks",
    ),
    "from_deepseek": ("instructor.v2.providers.openai.client", "from_deepseek"),
    "from_fireworks": (
        "instructor.v2.providers.fireworks.client",
        "from_fireworks",
    ),
    "from_gemini": ("instructor.v2.providers.gemini.client", "from_gemini"),
    "from_genai": ("instructor.v2.providers.genai.client", "from_genai"),
    "from_groq": ("instructor.v2.providers.groq.client", "from_groq"),
    "from_mistral": ("instructor.v2.providers.mistral.client", "from_mistral"),
    "from_openai": ("instructor.v2.providers.openai.client", "from_openai"),
    "from_openrouter": (
        "instructor.v2.providers.openrouter.client",
        "from_openrouter",
    ),
    "from_perplexity": (
        "instructor.v2.providers.perplexity.client",
        "from_perplexity",
    ),
    "from_together": ("instructor.v2.providers.openai.client", "from_together"),
    "from_vertexai": (
        "instructor.v2.providers.vertexai.client",
        "from_vertexai",
    ),
    "from_writer": ("instructor.v2.providers.writer.client", "from_writer"),
    "from_xai": ("instructor.v2.providers.xai.client", "from_xai"),
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_path)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
