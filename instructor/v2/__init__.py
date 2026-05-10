"""Instructor V2: Registry-based architecture.

This module provides the v2 implementation with a registry-based handler system.

Usage:
    from instructor import Mode
    from instructor.v2 import from_anthropic, from_openai, from_genai

    client = from_anthropic(anthropic_client, mode=Mode.TOOLS)
    client = from_openai(openai_client, mode=Mode.TOOLS)
    client = from_genai(genai_client, mode=Mode.TOOLS)
"""

import importlib
import importlib.util

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.protocols import ReaskHandler, RequestHandler, ResponseParser
from instructor.v2.core.registry import (
    ModeHandlers,
    ModeRegistry,
    mode_registry,
    normalize_mode,
)

# Keep the provider package reachable for public monkeypatch/import paths such
# as ``instructor.v2.providers.genai.client``.
providers = importlib.import_module("instructor.v2.providers")


def _lazy_import(module_path: str, func_name: str):
    def _wrapper(*args, **kwargs):
        module = importlib.import_module(module_path)
        return getattr(module, func_name)(*args, **kwargs)

    return _wrapper


def _maybe_export_client(func_name: str, module_path: str, sdk_module: str | None):
    if sdk_module:
        try:
            if importlib.util.find_spec(sdk_module) is None:
                return None
        except ModuleNotFoundError:
            return None
    return _lazy_import(module_path, func_name)


from_anthropic = _maybe_export_client(
    "from_anthropic",
    "instructor.v2.providers.anthropic.client",
    "anthropic",
)
from_openai = _maybe_export_client(
    "from_openai",
    "instructor.v2.providers.openai.client",
    "openai",
)
from_anyscale = _maybe_export_client(
    "from_anyscale",
    "instructor.v2.providers.openai.client",
    "openai",
)
from_together = _maybe_export_client(
    "from_together",
    "instructor.v2.providers.openai.client",
    "openai",
)
from_databricks = _maybe_export_client(
    "from_databricks",
    "instructor.v2.providers.openai.client",
    "openai",
)
from_deepseek = _maybe_export_client(
    "from_deepseek",
    "instructor.v2.providers.openai.client",
    "openai",
)
from_genai = _maybe_export_client(
    "from_genai",
    "instructor.v2.providers.genai.client",
    "google.genai",
)
from_gemini = _maybe_export_client(
    "from_gemini",
    "instructor.v2.providers.gemini.client",
    "google.generativeai",
)
from_cohere = _maybe_export_client(
    "from_cohere",
    "instructor.v2.providers.cohere.client",
    "cohere",
)
from_perplexity = _maybe_export_client(
    "from_perplexity",
    "instructor.v2.providers.perplexity.client",
    "openai",
)
from_mistral = _maybe_export_client(
    "from_mistral",
    "instructor.v2.providers.mistral.client",
    "mistralai",
)
from_openrouter = _maybe_export_client(
    "from_openrouter",
    "instructor.v2.providers.openrouter.client",
    "openai",
)
from_xai = _maybe_export_client(
    "from_xai",
    "instructor.v2.providers.xai.client",
    "xai_sdk",
)
from_groq = _maybe_export_client(
    "from_groq",
    "instructor.v2.providers.groq.client",
    "groq",
)
from_fireworks = _maybe_export_client(
    "from_fireworks",
    "instructor.v2.providers.fireworks.client",
    "fireworks",
)
from_cerebras = _maybe_export_client(
    "from_cerebras",
    "instructor.v2.providers.cerebras.client",
    "cerebras",
)
from_writer = _maybe_export_client(
    "from_writer",
    "instructor.v2.providers.writer.client",
    "writerai",
)
from_bedrock = _maybe_export_client(
    "from_bedrock",
    "instructor.v2.providers.bedrock.client",
    "botocore",
)
from_vertexai = _maybe_export_client(
    "from_vertexai",
    "instructor.v2.providers.vertexai.client",
    "vertexai",
)

_HANDLER_SPECS: dict[Provider, tuple[str, list[Mode]]] = {
    Provider.OPENAI: (
        "instructor.v2.providers.openai.handlers",
        [
            Mode.TOOLS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.PARALLEL_TOOLS,
            Mode.RESPONSES_TOOLS,
        ],
    ),
    Provider.ANYSCALE: (
        "instructor.v2.providers.openai.handlers",
        [
            Mode.TOOLS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.PARALLEL_TOOLS,
        ],
    ),
    Provider.TOGETHER: (
        "instructor.v2.providers.openai.handlers",
        [
            Mode.TOOLS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.PARALLEL_TOOLS,
        ],
    ),
    Provider.DATABRICKS: (
        "instructor.v2.providers.openai.handlers",
        [
            Mode.TOOLS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.PARALLEL_TOOLS,
        ],
    ),
    Provider.DEEPSEEK: (
        "instructor.v2.providers.openai.handlers",
        [
            Mode.TOOLS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.PARALLEL_TOOLS,
        ],
    ),
    Provider.OPENROUTER: (
        "instructor.v2.providers.openrouter.handlers",
        [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS],
    ),
    Provider.ANTHROPIC: (
        "instructor.v2.providers.anthropic.handlers",
        [
            Mode.TOOLS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.PARALLEL_TOOLS,
        ],
    ),
    Provider.GENAI: (
        "instructor.v2.providers.genai.handlers",
        [Mode.TOOLS, Mode.JSON],
    ),
    Provider.GEMINI: (
        "instructor.v2.providers.gemini.handlers",
        [Mode.TOOLS, Mode.MD_JSON],
    ),
    Provider.COHERE: (
        "instructor.v2.providers.cohere.handlers",
        [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
    ),
    Provider.PERPLEXITY: (
        "instructor.v2.providers.perplexity.handlers",
        [Mode.MD_JSON],
    ),
    Provider.XAI: (
        "instructor.v2.providers.xai.handlers",
        [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
    ),
    Provider.GROQ: (
        "instructor.v2.providers.openai.handlers",
        [Mode.TOOLS, Mode.MD_JSON],
    ),
    Provider.MISTRAL: (
        "instructor.v2.providers.mistral.handlers",
        [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
    ),
    Provider.VERTEXAI: (
        "instructor.v2.providers.vertexai.handlers",
        [Mode.TOOLS, Mode.MD_JSON, Mode.PARALLEL_TOOLS],
    ),
    Provider.FIREWORKS: (
        "instructor.v2.providers.openai.handlers",
        [Mode.TOOLS, Mode.MD_JSON],
    ),
    Provider.BEDROCK: (
        "instructor.v2.providers.bedrock.handlers",
        [Mode.TOOLS, Mode.MD_JSON],
    ),
    Provider.CEREBRAS: (
        "instructor.v2.providers.openai.handlers",
        [Mode.TOOLS, Mode.MD_JSON],
    ),
    Provider.WRITER: (
        "instructor.v2.providers.writer.handlers",
        [Mode.TOOLS, Mode.MD_JSON],
    ),
}


def _lazy_handler_loader(
    module_path: str, provider: Provider, mode: Mode
) -> ModeHandlers:
    importlib.import_module(module_path)
    return mode_registry._handlers[(provider, mode)]


for _provider, (_module_path, _modes) in _HANDLER_SPECS.items():
    for _mode in _modes:
        if mode_registry.is_registered(_provider, _mode):
            continue
        mode_registry.register_lazy(
            _provider,
            _mode,
            lambda mp=_module_path, p=_provider, m=_mode: _lazy_handler_loader(
                mp, p, m
            ),
        )

__all__ = [
    # Re-exports from instructor
    "Mode",
    "Provider",
    # Core infrastructure
    "ModeHandler",
    "ModeHandlers",
    "ModeRegistry",
    "mode_registry",
    "normalize_mode",
    "patch_v2",
    "register_mode_handler",
    # Protocols
    "ReaskHandler",
    "RequestHandler",
    "ResponseParser",
    # Providers
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


def patch_v2(*args, **kwargs):  # type: ignore[override]
    """Lazy import to avoid circular initialization during package import."""
    from instructor.v2.core.patch import patch_v2 as _patch_v2

    return _patch_v2(*args, **kwargs)
