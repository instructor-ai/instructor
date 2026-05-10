"""Shared provider capability matrix for v2 tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from instructor import Mode, Provider


@dataclass(frozen=True)
class ProviderSpec:
    provider_string: str | None
    supported_modes: tuple[Mode, ...]
    unsupported_modes: tuple[Mode, ...]
    legacy_modes: dict[Mode, Mode]
    from_function: str
    sdk_module: str
    basic_modes: tuple[Mode, ...] = ()
    async_modes: tuple[Mode, ...] = ()
    missing_sdk_message: str | None = None


PROVIDER_SPECS: dict[Provider, ProviderSpec] = {
    Provider.OPENAI: ProviderSpec(
        "openai/gpt-4o-mini",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
        (),
        {
            Mode.FUNCTIONS: Mode.TOOLS,
            Mode.TOOLS_STRICT: Mode.TOOLS,
            Mode.JSON_O1: Mode.JSON_SCHEMA,
        },
        "from_openai",
        "openai",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
    ),
    Provider.ANYSCALE: ProviderSpec(
        None,
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
        (Mode.RESPONSES_TOOLS,),
        {
            Mode.FUNCTIONS: Mode.TOOLS,
            Mode.TOOLS_STRICT: Mode.TOOLS,
            Mode.JSON_O1: Mode.JSON_SCHEMA,
        },
        "from_anyscale",
        "openai",
    ),
    Provider.TOGETHER: ProviderSpec(
        None,
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
        (Mode.RESPONSES_TOOLS,),
        {
            Mode.FUNCTIONS: Mode.TOOLS,
            Mode.TOOLS_STRICT: Mode.TOOLS,
            Mode.JSON_O1: Mode.JSON_SCHEMA,
        },
        "from_together",
        "openai",
    ),
    Provider.DATABRICKS: ProviderSpec(
        None,
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
        (Mode.RESPONSES_TOOLS,),
        {
            Mode.FUNCTIONS: Mode.TOOLS,
            Mode.TOOLS_STRICT: Mode.TOOLS,
            Mode.JSON_O1: Mode.JSON_SCHEMA,
        },
        "from_databricks",
        "openai",
    ),
    Provider.DEEPSEEK: ProviderSpec(
        None,
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
        (Mode.RESPONSES_TOOLS,),
        {
            Mode.FUNCTIONS: Mode.TOOLS,
            Mode.TOOLS_STRICT: Mode.TOOLS,
            Mode.JSON_O1: Mode.JSON_SCHEMA,
        },
        "from_deepseek",
        "openai",
    ),
    Provider.ANTHROPIC: ProviderSpec(
        "anthropic/claude-sonnet-4-6-20250627",
        (Mode.TOOLS, Mode.JSON, Mode.JSON_SCHEMA, Mode.PARALLEL_TOOLS),
        (Mode.MD_JSON,),
        {
            Mode.ANTHROPIC_TOOLS: Mode.TOOLS,
            Mode.ANTHROPIC_JSON: Mode.MD_JSON,
            Mode.ANTHROPIC_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
        },
        "from_anthropic",
        "anthropic",
        (Mode.TOOLS, Mode.JSON_SCHEMA),
        (Mode.TOOLS, Mode.JSON_SCHEMA),
    ),
    Provider.GENAI: ProviderSpec(
        "google/gemini-2.0-flash",
        (Mode.TOOLS, Mode.JSON),
        (Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
        {
            Mode.GENAI_TOOLS: Mode.TOOLS,
            Mode.GENAI_JSON: Mode.JSON,
            Mode.GENAI_STRUCTURED_OUTPUTS: Mode.JSON,
        },
        "from_genai",
        "google.genai",
        (Mode.TOOLS, Mode.JSON),
        (Mode.TOOLS, Mode.JSON),
    ),
    Provider.GEMINI: ProviderSpec(
        None,
        (Mode.TOOLS, Mode.MD_JSON),
        (Mode.JSON, Mode.JSON_SCHEMA, Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
        {Mode.GEMINI_TOOLS: Mode.TOOLS, Mode.GEMINI_JSON: Mode.MD_JSON},
        "from_gemini",
        "google.generativeai",
    ),
    Provider.COHERE: ProviderSpec(
        "cohere/command-a-03-2025",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
        {
            Mode.COHERE_TOOLS: Mode.TOOLS,
            Mode.COHERE_JSON_SCHEMA: Mode.JSON_SCHEMA,
        },
        "from_cohere",
        "cohere",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
    ),
    Provider.OPENROUTER: ProviderSpec(
        None,
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
        (Mode.RESPONSES_TOOLS,),
        {
            Mode.FUNCTIONS: Mode.TOOLS,
            Mode.TOOLS_STRICT: Mode.TOOLS,
            Mode.JSON_O1: Mode.JSON_SCHEMA,
            Mode.OPENROUTER_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
        },
        "from_openrouter",
        "openai",
    ),
    Provider.PERPLEXITY: ProviderSpec(
        None,
        (Mode.MD_JSON,),
        (Mode.JSON, Mode.TOOLS, Mode.JSON_SCHEMA, Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
        {Mode.PERPLEXITY_JSON: Mode.MD_JSON},
        "from_perplexity",
        "openai",
    ),
    Provider.XAI: ProviderSpec(
        "xai/grok-4.20-reasoning",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
        (Mode.RESPONSES_TOOLS,),
        {Mode.XAI_TOOLS: Mode.TOOLS, Mode.XAI_JSON: Mode.MD_JSON},
        "from_xai",
        "xai_sdk",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
    ),
    Provider.GROQ: ProviderSpec(
        "groq/llama-3.3-70b-versatile",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
        {},
        "from_groq",
        "groq",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
    ),
    Provider.MISTRAL: ProviderSpec(
        "mistral/ministral-8b-latest",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
        {
            Mode.MISTRAL_TOOLS: Mode.TOOLS,
            Mode.MISTRAL_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
        },
        "from_mistral",
        "mistralai",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
    ),
    Provider.FIREWORKS: ProviderSpec(
        "fireworks/accounts/fireworks/models/kimi-k2p5",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
        {Mode.FIREWORKS_TOOLS: Mode.TOOLS, Mode.FIREWORKS_JSON: Mode.MD_JSON},
        "from_fireworks",
        "fireworks",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
    ),
    Provider.CEREBRAS: ProviderSpec(
        "cerebras/gpt-oss-120b",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
        (Mode.RESPONSES_TOOLS,),
        {Mode.CEREBRAS_TOOLS: Mode.TOOLS, Mode.CEREBRAS_JSON: Mode.MD_JSON},
        "from_cerebras",
        "cerebras.cloud.sdk",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        "cerebras is not installed",
    ),
    Provider.WRITER: ProviderSpec(
        "writer/palmyra-x5",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
        {Mode.WRITER_TOOLS: Mode.TOOLS, Mode.WRITER_JSON: Mode.MD_JSON},
        "from_writer",
        "writerai",
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        (Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
    ),
    Provider.BEDROCK: ProviderSpec(
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        (Mode.TOOLS, Mode.MD_JSON),
        (Mode.JSON_SCHEMA, Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
        {Mode.BEDROCK_TOOLS: Mode.TOOLS, Mode.BEDROCK_JSON: Mode.MD_JSON},
        "from_bedrock",
        "botocore",
        (Mode.TOOLS, Mode.MD_JSON),
        (Mode.TOOLS, Mode.MD_JSON),
    ),
    Provider.VERTEXAI: ProviderSpec(
        None,
        (Mode.TOOLS, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
        (Mode.JSON, Mode.JSON_SCHEMA, Mode.RESPONSES_TOOLS),
        {
            Mode.VERTEXAI_TOOLS: Mode.TOOLS,
            Mode.VERTEXAI_JSON: Mode.MD_JSON,
            Mode.VERTEXAI_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
        },
        "from_vertexai",
        "vertexai",
    ),
}


PROVIDER_HANDLER_MODES: dict[Provider, tuple[Mode, ...]] = {
    provider: spec.supported_modes for provider, spec in PROVIDER_SPECS.items()
}


def legacy_config_dicts() -> dict[Provider, dict[str, Any]]:
    """Expose the old dict shape while tests migrate to ProviderSpec."""
    return {
        provider: {
            "provider_string": spec.provider_string,
            "supported_modes": list(spec.supported_modes),
            "unsupported_modes": list(spec.unsupported_modes),
            "legacy_modes": spec.legacy_modes,
            "from_function": spec.from_function,
            "sdk_module": spec.sdk_module,
            "basic_modes": list(spec.basic_modes),
            "async_modes": list(spec.async_modes),
            "missing_sdk_message": spec.missing_sdk_message,
        }
        for provider, spec in PROVIDER_SPECS.items()
    }
