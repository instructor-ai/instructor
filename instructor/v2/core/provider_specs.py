"""Single source of truth for v2 provider capabilities and public bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider


@dataclass(frozen=True)
class ProviderCapabilities:
    """User-visible provider features backed by deterministic conformance tests.

    ``multimodal_inputs`` describes typed v2 media that Instructor converts into
    provider wire formats, not media a user may pre-encode for an SDK directly.
    """

    partial_stream_modes: tuple[Mode, ...] = ()
    iterable_stream_modes: tuple[Mode, ...] = ()
    multimodal_inputs: tuple[str, ...] = ()
    explicit_parallel_tools: bool = False


_NO_CAPABILITIES = ProviderCapabilities()


@dataclass(frozen=True)
class ClientSpec:
    """Declarative native-client contract used by the shared factory runtime."""

    sync_types: tuple[str, ...]
    create: str
    async_types: tuple[str, ...] = ()
    async_create: str | None = None
    stream: str | None = None
    async_stream: str | None = None
    falsey_model_fallback: bool = False
    validation_order: Literal["dependency-first", "mode-first", "client-first"] = (
        "dependency-first"
    )


@dataclass(frozen=True)
class ProviderSpec:
    provider: Provider
    canonical_provider: Provider
    aliases: tuple[str, ...]
    handler_module: str | None
    supported_modes: tuple[Mode, ...]
    unsupported_modes: tuple[Mode, ...]
    legacy_modes: Mapping[Mode, Mode]
    from_function: str | None
    client_module: str | None
    sdk_module: str | None
    capabilities: ProviderCapabilities = _NO_CAPABILITIES
    builder_module: str | None = None
    provider_string: str | None = None
    basic_modes: tuple[Mode, ...] = ()
    async_modes: tuple[Mode, ...] = ()
    missing_sdk_message: str | None = None
    client: ClientSpec | None = None

    @property
    def model_builder_module(self) -> str | None:
        """Resolve the lazily imported model builder module by convention."""
        return self.builder_module or self.client_module


def _spec(
    provider: Provider,
    *,
    aliases: tuple[str, ...],
    canonical_provider: Provider | None = None,
    handler_module: str | None,
    supported_modes: tuple[Mode, ...],
    unsupported_modes: tuple[Mode, ...],
    legacy_modes: dict[Mode, Mode],
    from_function: str | None,
    client_module: str | None,
    sdk_module: str | None,
    capabilities: ProviderCapabilities = _NO_CAPABILITIES,
    builder_module: str | None = None,
    provider_string: str | None = None,
    basic_modes: tuple[Mode, ...] = (),
    async_modes: tuple[Mode, ...] = (),
    missing_sdk_message: str | None = None,
    client: ClientSpec | None = None,
) -> ProviderSpec:
    return ProviderSpec(
        provider=provider,
        canonical_provider=canonical_provider or provider,
        aliases=aliases,
        handler_module=handler_module,
        supported_modes=supported_modes,
        unsupported_modes=unsupported_modes,
        legacy_modes=MappingProxyType(legacy_modes),
        from_function=from_function,
        client_module=client_module,
        sdk_module=sdk_module,
        capabilities=capabilities,
        builder_module=builder_module,
        provider_string=provider_string,
        basic_modes=basic_modes,
        async_modes=async_modes,
        missing_sdk_message=missing_sdk_message,
        client=client,
    )


_OPENAI_COMPAT_SUPPORTED_MODES = (
    Mode.TOOLS,
    Mode.JSON,
    Mode.JSON_SCHEMA,
    Mode.MD_JSON,
    Mode.PARALLEL_TOOLS,
)
_OPENAI_COMPAT_UNSUPPORTED_MODES = (Mode.RESPONSES_TOOLS,)
_OPENAI_COMPAT_LEGACY_MODES = {
    Mode.FUNCTIONS: Mode.TOOLS,
    Mode.TOOLS_STRICT: Mode.TOOLS,
    Mode.JSON_O1: Mode.JSON_SCHEMA,
}
_OPENAI_MULTIMODAL_INPUTS = ("image", "audio", "pdf")
_OPENAI_COMPAT_CAPABILITIES = ProviderCapabilities(
    partial_stream_modes=(Mode.TOOLS, Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON),
    iterable_stream_modes=(Mode.TOOLS,),
    multimodal_inputs=_OPENAI_MULTIMODAL_INPUTS,
    explicit_parallel_tools=True,
)
_OPENAI_CLIENT = ClientSpec(
    sync_types=("openai.OpenAI", "openai.AzureOpenAI"),
    async_types=("openai.AsyncOpenAI", "openai.AsyncAzureOpenAI"),
    create="chat.completions.create",
)


def _openai_compat_spec(
    provider: Provider,
    *,
    alias: str,
    from_function: str,
) -> ProviderSpec:
    return _spec(
        provider,
        aliases=(alias,),
        handler_module="instructor.v2.providers.openai.handlers",
        supported_modes=_OPENAI_COMPAT_SUPPORTED_MODES,
        unsupported_modes=_OPENAI_COMPAT_UNSUPPORTED_MODES,
        legacy_modes=_OPENAI_COMPAT_LEGACY_MODES,
        from_function=from_function,
        client_module="instructor.v2.providers.openai.client",
        sdk_module="openai",
        capabilities=_OPENAI_COMPAT_CAPABILITIES,
        client=_OPENAI_CLIENT,
    )


PROVIDER_SPECS: Mapping[Provider, ProviderSpec] = MappingProxyType(
    {
        Provider.OPENAI: _spec(
            Provider.OPENAI,
            aliases=("openai",),
            handler_module="instructor.v2.providers.openai.handlers",
            supported_modes=(
                Mode.TOOLS,
                Mode.JSON,
                Mode.JSON_SCHEMA,
                Mode.MD_JSON,
                Mode.PARALLEL_TOOLS,
                Mode.RESPONSES_TOOLS,
            ),
            unsupported_modes=(),
            legacy_modes={
                Mode.FUNCTIONS: Mode.TOOLS,
                Mode.TOOLS_STRICT: Mode.TOOLS,
                Mode.JSON_O1: Mode.JSON_SCHEMA,
                Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: Mode.RESPONSES_TOOLS,
            },
            from_function="from_openai",
            client_module="instructor.v2.providers.openai.client",
            sdk_module="openai",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(
                    Mode.TOOLS,
                    Mode.JSON,
                    Mode.JSON_SCHEMA,
                    Mode.MD_JSON,
                    Mode.RESPONSES_TOOLS,
                ),
                iterable_stream_modes=(Mode.TOOLS, Mode.RESPONSES_TOOLS),
                multimodal_inputs=_OPENAI_MULTIMODAL_INPUTS,
                explicit_parallel_tools=True,
            ),
            provider_string="openai/gpt-4o-mini",
            basic_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            async_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            client=_OPENAI_CLIENT,
        ),
        Provider.ANYSCALE: _openai_compat_spec(
            Provider.ANYSCALE,
            alias="anyscale",
            from_function="from_anyscale",
        ),
        Provider.TOGETHER: _openai_compat_spec(
            Provider.TOGETHER,
            alias="together",
            from_function="from_together",
        ),
        Provider.DATABRICKS: _openai_compat_spec(
            Provider.DATABRICKS,
            alias="databricks",
            from_function="from_databricks",
        ),
        Provider.DEEPSEEK: _openai_compat_spec(
            Provider.DEEPSEEK,
            alias="deepseek",
            from_function="from_deepseek",
        ),
        Provider.OPENROUTER: _spec(
            Provider.OPENROUTER,
            aliases=("openrouter",),
            handler_module="instructor.v2.providers.openrouter.handlers",
            supported_modes=(
                Mode.TOOLS,
                Mode.JSON_SCHEMA,
                Mode.MD_JSON,
                Mode.PARALLEL_TOOLS,
            ),
            unsupported_modes=(Mode.RESPONSES_TOOLS,),
            legacy_modes={
                Mode.FUNCTIONS: Mode.TOOLS,
                Mode.TOOLS_STRICT: Mode.TOOLS,
                Mode.JSON_O1: Mode.JSON_SCHEMA,
                Mode.OPENROUTER_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
            },
            from_function="from_openrouter",
            client_module="instructor.v2.providers.openrouter.client",
            sdk_module="openai",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
                iterable_stream_modes=(Mode.TOOLS,),
                multimodal_inputs=_OPENAI_MULTIMODAL_INPUTS,
                explicit_parallel_tools=True,
            ),
            client=_OPENAI_CLIENT,
        ),
        Provider.ANTHROPIC: _spec(
            Provider.ANTHROPIC,
            aliases=("anthropic",),
            handler_module="instructor.v2.providers.anthropic.handlers",
            supported_modes=(
                Mode.TOOLS,
                Mode.JSON,
                Mode.JSON_SCHEMA,
                Mode.PARALLEL_TOOLS,
            ),
            unsupported_modes=(Mode.MD_JSON,),
            legacy_modes={
                Mode.ANTHROPIC_TOOLS: Mode.TOOLS,
                Mode.ANTHROPIC_REASONING_TOOLS: Mode.TOOLS,
                Mode.ANTHROPIC_JSON: Mode.JSON,
                Mode.ANTHROPIC_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
            },
            from_function="from_anthropic",
            client_module="instructor.v2.providers.anthropic.client",
            sdk_module="anthropic",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS, Mode.JSON, Mode.JSON_SCHEMA),
                iterable_stream_modes=(Mode.TOOLS,),
                multimodal_inputs=("image", "pdf"),
                explicit_parallel_tools=True,
            ),
            provider_string="anthropic/claude-sonnet-4-6",
            basic_modes=(Mode.TOOLS, Mode.JSON_SCHEMA),
            async_modes=(Mode.TOOLS, Mode.JSON_SCHEMA),
            missing_sdk_message=(
                "anthropic is not installed. Install it with: pip install anthropic"
            ),
            client=ClientSpec(
                sync_types=(
                    "anthropic.Anthropic",
                    "anthropic.AnthropicBedrock",
                    "anthropic.AnthropicVertex",
                ),
                async_types=(
                    "anthropic.AsyncAnthropic",
                    "anthropic.AsyncAnthropicBedrock",
                    "anthropic.AsyncAnthropicVertex",
                ),
                create="messages.create",
            ),
        ),
        Provider.GENAI: _spec(
            Provider.GENAI,
            aliases=("google",),
            handler_module="instructor.v2.providers.genai.handlers",
            supported_modes=(Mode.TOOLS, Mode.JSON),
            unsupported_modes=(Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
            legacy_modes={
                Mode.GENAI_TOOLS: Mode.TOOLS,
                Mode.GENAI_JSON: Mode.JSON,
                Mode.GENAI_STRUCTURED_OUTPUTS: Mode.JSON,
            },
            from_function="from_genai",
            client_module="instructor.v2.providers.genai.client",
            sdk_module="google.genai",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS, Mode.JSON),
                iterable_stream_modes=(Mode.TOOLS, Mode.JSON),
                multimodal_inputs=("image", "audio", "pdf"),
            ),
            provider_string="google/gemini-3.5-flash",
            basic_modes=(Mode.TOOLS, Mode.JSON),
            async_modes=(Mode.TOOLS, Mode.JSON),
            missing_sdk_message=(
                "google-genai is not installed. Install it with: pip install google-genai"
            ),
            client=ClientSpec(
                sync_types=("google.genai.Client",),
                create="models.generate_content",
                async_create="aio.models.generate_content",
                stream="models.generate_content_stream",
                async_stream="aio.models.generate_content_stream",
                falsey_model_fallback=True,
                validation_order="client-first",
            ),
        ),
        Provider.GENERATIVE_AI: _spec(
            Provider.GENERATIVE_AI,
            aliases=("generative-ai",),
            canonical_provider=Provider.GENAI,
            handler_module=None,
            supported_modes=(),
            unsupported_modes=(),
            legacy_modes={
                Mode.GENAI_TOOLS: Mode.TOOLS,
                Mode.GENAI_JSON: Mode.JSON,
                Mode.GENAI_STRUCTURED_OUTPUTS: Mode.JSON,
            },
            from_function=None,
            client_module=None,
            sdk_module="google.genai",
            builder_module="instructor.v2.providers.genai.client",
        ),
        Provider.GEMINI: _spec(
            Provider.GEMINI,
            aliases=("gemini",),
            handler_module="instructor.v2.providers.gemini.handlers",
            supported_modes=(Mode.TOOLS, Mode.MD_JSON),
            unsupported_modes=(
                Mode.JSON,
                Mode.JSON_SCHEMA,
                Mode.PARALLEL_TOOLS,
                Mode.RESPONSES_TOOLS,
            ),
            legacy_modes={
                Mode.GEMINI_TOOLS: Mode.TOOLS,
                Mode.GEMINI_JSON: Mode.MD_JSON,
            },
            from_function="from_gemini",
            client_module="instructor.v2.providers.gemini.client",
            sdk_module="google.generativeai",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS, Mode.MD_JSON),
            ),
            missing_sdk_message=(
                "google-generativeai is not installed. Install it with: "
                "pip install google-generativeai"
            ),
            client=ClientSpec(
                sync_types=("google.generativeai.GenerativeModel",),
                create="generate_content",
                async_create="generate_content_async",
                validation_order="mode-first",
            ),
        ),
        Provider.COHERE: _spec(
            Provider.COHERE,
            aliases=("cohere",),
            handler_module="instructor.v2.providers.cohere.handlers",
            supported_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            unsupported_modes=(Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
            legacy_modes={
                Mode.COHERE_TOOLS: Mode.TOOLS,
                Mode.COHERE_JSON_SCHEMA: Mode.JSON_SCHEMA,
            },
            from_function="from_cohere",
            client_module="instructor.v2.providers.cohere.client",
            sdk_module="cohere",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS,),
                iterable_stream_modes=(Mode.TOOLS,),
            ),
            provider_string="cohere/command-a-03-2025",
            basic_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            async_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            missing_sdk_message=(
                "cohere is not installed. Install it with: pip install cohere"
            ),
            client=ClientSpec(
                sync_types=("cohere.Client", "cohere.ClientV2"),
                async_types=("cohere.AsyncClient", "cohere.AsyncClientV2"),
                create="chat",
                stream="chat_stream",
                async_stream="chat_stream",
            ),
        ),
        Provider.PERPLEXITY: _spec(
            Provider.PERPLEXITY,
            aliases=("perplexity",),
            handler_module="instructor.v2.providers.perplexity.handlers",
            supported_modes=(Mode.MD_JSON,),
            unsupported_modes=(
                Mode.JSON,
                Mode.TOOLS,
                Mode.JSON_SCHEMA,
                Mode.PARALLEL_TOOLS,
                Mode.RESPONSES_TOOLS,
            ),
            legacy_modes={Mode.PERPLEXITY_JSON: Mode.MD_JSON},
            from_function="from_perplexity",
            client_module="instructor.v2.providers.perplexity.client",
            sdk_module="openai",
            capabilities=ProviderCapabilities(
                multimodal_inputs=_OPENAI_MULTIMODAL_INPUTS,
            ),
            client=_OPENAI_CLIENT,
        ),
        Provider.XAI: _spec(
            Provider.XAI,
            aliases=("xai",),
            handler_module="instructor.v2.providers.xai.handlers",
            supported_modes=(
                Mode.TOOLS,
                Mode.JSON_SCHEMA,
                Mode.MD_JSON,
                Mode.PARALLEL_TOOLS,
            ),
            unsupported_modes=(Mode.RESPONSES_TOOLS,),
            legacy_modes={Mode.XAI_TOOLS: Mode.TOOLS, Mode.XAI_JSON: Mode.MD_JSON},
            from_function="from_xai",
            client_module="instructor.v2.providers.xai.client",
            sdk_module="xai_sdk",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS, Mode.JSON_SCHEMA),
                iterable_stream_modes=(Mode.TOOLS, Mode.JSON_SCHEMA),
                explicit_parallel_tools=True,
            ),
            provider_string="xai/grok-4.20-reasoning",
            basic_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            async_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
        ),
        Provider.GROQ: _spec(
            Provider.GROQ,
            aliases=("groq",),
            handler_module="instructor.v2.providers.openai.handlers",
            supported_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            unsupported_modes=(Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
            legacy_modes={},
            from_function="from_groq",
            client_module="instructor.v2.providers.groq.client",
            sdk_module="groq",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
                iterable_stream_modes=(Mode.TOOLS,),
                multimodal_inputs=_OPENAI_MULTIMODAL_INPUTS,
            ),
            provider_string="groq/llama-3.3-70b-versatile",
            basic_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            async_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            missing_sdk_message=(
                "groq is not installed. Install it with: pip install groq"
            ),
            client=ClientSpec(
                sync_types=("groq.Groq",),
                async_types=("groq.AsyncGroq",),
                create="chat.completions.create",
            ),
        ),
        Provider.MISTRAL: _spec(
            Provider.MISTRAL,
            aliases=("mistral",),
            handler_module="instructor.v2.providers.mistral.handlers",
            supported_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            unsupported_modes=(Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
            legacy_modes={
                Mode.MISTRAL_TOOLS: Mode.TOOLS,
                Mode.MISTRAL_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
            },
            from_function="from_mistral",
            client_module="instructor.v2.providers.mistral.client",
            sdk_module="mistralai",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
                iterable_stream_modes=(Mode.TOOLS,),
                multimodal_inputs=("image", "audio", "pdf"),
            ),
            provider_string="mistral/ministral-8b-latest",
            basic_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            async_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            missing_sdk_message=(
                "mistralai is not installed. Install it with: pip install mistralai"
            ),
            client=ClientSpec(
                sync_types=("mistralai.Mistral",),
                create="chat.complete",
                async_create="chat.complete_async",
                stream="chat.stream",
                async_stream="chat.stream_async",
            ),
        ),
        Provider.FIREWORKS: _spec(
            Provider.FIREWORKS,
            aliases=("fireworks",),
            handler_module="instructor.v2.providers.openai.handlers",
            supported_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            unsupported_modes=(Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
            legacy_modes={
                Mode.FIREWORKS_TOOLS: Mode.TOOLS,
                Mode.FIREWORKS_JSON: Mode.MD_JSON,
            },
            from_function="from_fireworks",
            client_module="instructor.v2.providers.fireworks.client",
            sdk_module="fireworks",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
                iterable_stream_modes=(Mode.TOOLS,),
                multimodal_inputs=_OPENAI_MULTIMODAL_INPUTS,
            ),
            provider_string="fireworks/accounts/fireworks/models/kimi-k2p5",
            basic_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            async_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            missing_sdk_message=(
                "fireworks is not installed. Install it with: pip install fireworks-ai"
            ),
            client=ClientSpec(
                sync_types=("fireworks.client.Fireworks",),
                async_types=("fireworks.client.AsyncFireworks",),
                create="chat.completions.create",
                async_create="chat.completions.acreate",
            ),
        ),
        Provider.CEREBRAS: _spec(
            Provider.CEREBRAS,
            aliases=("cerebras",),
            handler_module="instructor.v2.providers.openai.handlers",
            supported_modes=(
                Mode.TOOLS,
                Mode.JSON_SCHEMA,
                Mode.MD_JSON,
                Mode.PARALLEL_TOOLS,
            ),
            unsupported_modes=(Mode.RESPONSES_TOOLS,),
            legacy_modes={
                Mode.CEREBRAS_TOOLS: Mode.TOOLS,
                Mode.CEREBRAS_JSON: Mode.MD_JSON,
            },
            from_function="from_cerebras",
            client_module="instructor.v2.providers.cerebras.client",
            sdk_module="cerebras.cloud.sdk",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
                iterable_stream_modes=(Mode.TOOLS,),
                multimodal_inputs=_OPENAI_MULTIMODAL_INPUTS,
                explicit_parallel_tools=True,
            ),
            provider_string="cerebras/gpt-oss-120b",
            basic_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            async_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            missing_sdk_message=(
                "cerebras is not installed. Install it with: "
                "pip install cerebras-cloud-sdk"
            ),
            client=ClientSpec(
                sync_types=("cerebras.cloud.sdk.Cerebras",),
                async_types=("cerebras.cloud.sdk.AsyncCerebras",),
                create="chat.completions.create",
            ),
        ),
        Provider.WRITER: _spec(
            Provider.WRITER,
            aliases=("writer",),
            handler_module="instructor.v2.providers.writer.handlers",
            supported_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            unsupported_modes=(Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS),
            legacy_modes={
                Mode.WRITER_TOOLS: Mode.TOOLS,
                Mode.WRITER_JSON: Mode.MD_JSON,
            },
            from_function="from_writer",
            client_module="instructor.v2.providers.writer.client",
            sdk_module="writerai",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
                iterable_stream_modes=(Mode.TOOLS,),
                multimodal_inputs=_OPENAI_MULTIMODAL_INPUTS,
            ),
            provider_string="writer/palmyra-x5",
            basic_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            async_modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON),
            missing_sdk_message=(
                "writerai is not installed. Install it with: pip install writer-sdk"
            ),
            client=ClientSpec(
                sync_types=("writerai.Writer",),
                async_types=("writerai.AsyncWriter",),
                create="chat.chat",
            ),
        ),
        Provider.BEDROCK: _spec(
            Provider.BEDROCK,
            aliases=("bedrock",),
            handler_module="instructor.v2.providers.bedrock.handlers",
            supported_modes=(Mode.TOOLS, Mode.MD_JSON),
            unsupported_modes=(
                Mode.JSON_SCHEMA,
                Mode.PARALLEL_TOOLS,
                Mode.RESPONSES_TOOLS,
            ),
            legacy_modes={
                Mode.BEDROCK_TOOLS: Mode.TOOLS,
                Mode.BEDROCK_JSON: Mode.MD_JSON,
            },
            from_function="from_bedrock",
            client_module="instructor.v2.providers.bedrock.client",
            sdk_module="botocore",
            provider_string="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            basic_modes=(Mode.TOOLS, Mode.MD_JSON),
            async_modes=(Mode.TOOLS, Mode.MD_JSON),
            missing_sdk_message=(
                "botocore is not installed. Install it with: pip install boto3"
            ),
            client=ClientSpec(
                sync_types=("botocore.client.BaseClient",),
                create="converse",
            ),
        ),
        Provider.VERTEXAI: _spec(
            Provider.VERTEXAI,
            aliases=("vertexai",),
            handler_module="instructor.v2.providers.vertexai.handlers",
            supported_modes=(Mode.TOOLS, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
            unsupported_modes=(Mode.JSON, Mode.JSON_SCHEMA, Mode.RESPONSES_TOOLS),
            legacy_modes={
                Mode.VERTEXAI_TOOLS: Mode.TOOLS,
                Mode.VERTEXAI_JSON: Mode.MD_JSON,
                Mode.VERTEXAI_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
            },
            from_function="from_vertexai",
            client_module="instructor.v2.providers.vertexai.client",
            sdk_module="vertexai",
            capabilities=ProviderCapabilities(
                partial_stream_modes=(Mode.TOOLS, Mode.MD_JSON),
                explicit_parallel_tools=True,
            ),
            missing_sdk_message=(
                "vertexai is not installed. Install it with: "
                "pip install google-cloud-aiplatform"
            ),
            client=ClientSpec(
                sync_types=("vertexai.generative_models.GenerativeModel",),
                create="generate_content",
                async_create="generate_content_async",
                validation_order="mode-first",
            ),
        ),
        Provider.AZURE_OPENAI: _spec(
            Provider.AZURE_OPENAI,
            aliases=("azure_openai",),
            canonical_provider=Provider.OPENAI,
            handler_module=None,
            supported_modes=(),
            unsupported_modes=(),
            legacy_modes={},
            from_function=None,
            client_module=None,
            sdk_module="openai",
            builder_module="instructor.v2.providers.openai.client",
        ),
        Provider.OLLAMA: _spec(
            Provider.OLLAMA,
            aliases=("ollama",),
            canonical_provider=Provider.OPENAI,
            handler_module=None,
            supported_modes=(),
            unsupported_modes=(),
            legacy_modes={},
            from_function=None,
            client_module=None,
            sdk_module="openai",
            builder_module="instructor.v2.providers.openai.client",
        ),
        Provider.LITELLM: _spec(
            Provider.LITELLM,
            aliases=("litellm",),
            handler_module=None,
            supported_modes=(),
            unsupported_modes=(),
            legacy_modes={},
            from_function="from_litellm",
            client_module="instructor.v2.providers.litellm.client",
            sdk_module=None,
        ),
    }
)

ALIAS_TO_PROVIDER: Mapping[str, Provider] = MappingProxyType(
    {
        alias: provider
        for provider, spec in PROVIDER_SPECS.items()
        for alias in spec.aliases
    }
)

PUBLIC_FACTORY_ATTRS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        spec.from_function: (spec.client_module, spec.from_function)
        for spec in PROVIDER_SPECS.values()
        if spec.from_function is not None and spec.client_module is not None
    }
)

HANDLER_SPECS: Mapping[Provider, tuple[str, tuple[Mode, ...]]] = MappingProxyType(
    {
        provider: (spec.handler_module, spec.supported_modes)
        for provider, spec in PROVIDER_SPECS.items()
        if spec.handler_module is not None and spec.supported_modes
    }
)
