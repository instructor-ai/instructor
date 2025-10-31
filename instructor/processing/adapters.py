from __future__ import annotations

"""Provider specific adapters used by the response processing pipeline."""

from dataclasses import dataclass
from typing import Any, Callable

from ..mode import Mode
from ..providers.anthropic.utils import (
    handle_anthropic_json,
    handle_anthropic_parallel_tools,
    handle_anthropic_reasoning_tools,
    handle_anthropic_tools,
    reask_anthropic_json,
    reask_anthropic_tools,
)
from ..providers.bedrock.utils import (
    handle_bedrock_json,
    handle_bedrock_tools,
    reask_bedrock_json,
    reask_bedrock_tools,
)
from ..providers.cerebras.utils import (
    handle_cerebras_json,
    handle_cerebras_tools,
    reask_cerebras_tools,
)
from ..providers.cohere.utils import (
    handle_cohere_json_schema,
    handle_cohere_tools,
    reask_cohere_tools,
)
from ..providers.fireworks.utils import (
    handle_fireworks_json,
    handle_fireworks_tools,
    reask_fireworks_json,
    reask_fireworks_tools,
)
from ..providers.gemini.utils import (
    handle_gemini_json,
    handle_gemini_tools,
    handle_genai_structured_outputs,
    handle_genai_tools,
    handle_vertexai_json,
    handle_vertexai_parallel_tools,
    handle_vertexai_tools,
    reask_gemini_json,
    reask_gemini_tools,
    reask_genai_structured_outputs,
    reask_genai_tools,
    reask_vertexai_json,
    reask_vertexai_tools,
)
from ..providers.mistral.utils import (
    handle_mistral_structured_outputs,
    handle_mistral_tools,
    reask_mistral_structured_outputs,
    reask_mistral_tools,
)
from ..providers.openai.utils import (
    handle_functions,
    handle_json_modes,
    handle_json_o1,
    handle_openrouter_structured_outputs,
    handle_parallel_tools,
    handle_responses_tools,
    handle_responses_tools_with_inbuilt_tools,
    handle_tools,
    handle_tools_strict,
    reask_default,
    reask_md_json,
    reask_responses_tools,
    reask_tools,
)
from ..providers.perplexity.utils import (
    handle_perplexity_json,
    reask_perplexity_json,
)
from ..providers.writer.utils import (
    handle_writer_json,
    handle_writer_tools,
    reask_writer_json,
    reask_writer_tools,
)
from ..providers.xai.utils import (
    handle_xai_json,
    handle_xai_tools,
    reask_xai_json,
    reask_xai_tools,
)


PrepareRequestCallable = Callable[
    [type[Any] | None, dict[str, Any], "ProviderAdapterContext"],
    tuple[type[Any] | None, dict[str, Any]],
]

ReaskCallable = Callable[[dict[str, Any], Any, Exception], dict[str, Any]]


@dataclass(frozen=True)
class ProviderAdapterContext:
    """Additional data passed to provider adapters."""

    mode: Mode
    autodetect_images: bool = False


@dataclass(frozen=True)
class ProviderAdapter:
    """Collection of provider specific hooks for request/response handling."""

    prepare_request: PrepareRequestCallable
    reask: ReaskCallable
    prepare_model: bool = True
    convert_messages: bool = True


def _identity_prepare(
    fn: Callable[[type[Any] | None, dict[str, Any]], tuple[type[Any] | None, dict[str, Any]]]
) -> PrepareRequestCallable:
    return lambda response_model, kwargs, _context: fn(response_model, kwargs)


PROVIDER_ADAPTERS: dict[Mode, ProviderAdapter] = {
    Mode.PARALLEL_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_parallel_tools),
        reask=reask_tools,
        prepare_model=False,
        convert_messages=False,
    ),
    Mode.VERTEXAI_PARALLEL_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_vertexai_parallel_tools),
        reask=reask_vertexai_tools,
        prepare_model=False,
        convert_messages=False,
    ),
    Mode.ANTHROPIC_PARALLEL_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_anthropic_parallel_tools),
        reask=reask_anthropic_tools,
        prepare_model=False,
        convert_messages=False,
    ),
    Mode.FUNCTIONS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_functions),
        reask=reask_default,
    ),
    Mode.TOOLS_STRICT: ProviderAdapter(
        prepare_request=_identity_prepare(handle_tools_strict),
        reask=reask_tools,
    ),
    Mode.TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_tools),
        reask=reask_tools,
    ),
    Mode.MISTRAL_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_mistral_tools),
        reask=reask_mistral_tools,
    ),
    Mode.MISTRAL_STRUCTURED_OUTPUTS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_mistral_structured_outputs),
        reask=reask_mistral_structured_outputs,
    ),
    Mode.JSON_O1: ProviderAdapter(
        prepare_request=_identity_prepare(handle_json_o1),
        reask=reask_default,
    ),
    Mode.JSON: ProviderAdapter(
        prepare_request=lambda rm, kw, ctx: handle_json_modes(rm, kw, ctx.mode),
        reask=reask_md_json,
    ),
    Mode.MD_JSON: ProviderAdapter(
        prepare_request=lambda rm, kw, ctx: handle_json_modes(rm, kw, ctx.mode),
        reask=reask_md_json,
    ),
    Mode.JSON_SCHEMA: ProviderAdapter(
        prepare_request=lambda rm, kw, ctx: handle_json_modes(rm, kw, ctx.mode),
        reask=reask_md_json,
    ),
    Mode.ANTHROPIC_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_anthropic_tools),
        reask=reask_anthropic_tools,
    ),
    Mode.ANTHROPIC_REASONING_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_anthropic_reasoning_tools),
        reask=reask_anthropic_tools,
    ),
    Mode.ANTHROPIC_JSON: ProviderAdapter(
        prepare_request=_identity_prepare(handle_anthropic_json),
        reask=reask_anthropic_json,
    ),
    Mode.COHERE_JSON_SCHEMA: ProviderAdapter(
        prepare_request=_identity_prepare(handle_cohere_json_schema),
        reask=reask_cohere_tools,
    ),
    Mode.COHERE_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_cohere_tools),
        reask=reask_cohere_tools,
    ),
    Mode.GEMINI_JSON: ProviderAdapter(
        prepare_request=_identity_prepare(handle_gemini_json),
        reask=reask_gemini_json,
    ),
    Mode.GEMINI_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_gemini_tools),
        reask=reask_gemini_tools,
    ),
    Mode.GENAI_TOOLS: ProviderAdapter(
        prepare_request=lambda rm, kw, ctx: handle_genai_tools(
            rm, kw, ctx.autodetect_images
        ),
        reask=reask_genai_tools,
    ),
    Mode.GENAI_STRUCTURED_OUTPUTS: ProviderAdapter(
        prepare_request=lambda rm, kw, ctx: handle_genai_structured_outputs(
            rm, kw, ctx.autodetect_images
        ),
        reask=reask_genai_structured_outputs,
    ),
    Mode.VERTEXAI_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_vertexai_tools),
        reask=reask_vertexai_tools,
    ),
    Mode.VERTEXAI_JSON: ProviderAdapter(
        prepare_request=_identity_prepare(handle_vertexai_json),
        reask=reask_vertexai_json,
    ),
    Mode.CEREBRAS_JSON: ProviderAdapter(
        prepare_request=_identity_prepare(handle_cerebras_json),
        reask=reask_default,
    ),
    Mode.CEREBRAS_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_cerebras_tools),
        reask=reask_cerebras_tools,
    ),
    Mode.FIREWORKS_JSON: ProviderAdapter(
        prepare_request=_identity_prepare(handle_fireworks_json),
        reask=reask_fireworks_json,
    ),
    Mode.FIREWORKS_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_fireworks_tools),
        reask=reask_fireworks_tools,
    ),
    Mode.WRITER_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_writer_tools),
        reask=reask_writer_tools,
    ),
    Mode.WRITER_JSON: ProviderAdapter(
        prepare_request=_identity_prepare(handle_writer_json),
        reask=reask_writer_json,
    ),
    Mode.BEDROCK_JSON: ProviderAdapter(
        prepare_request=_identity_prepare(handle_bedrock_json),
        reask=reask_bedrock_json,
    ),
    Mode.BEDROCK_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_bedrock_tools),
        reask=reask_bedrock_tools,
    ),
    Mode.PERPLEXITY_JSON: ProviderAdapter(
        prepare_request=_identity_prepare(handle_perplexity_json),
        reask=reask_perplexity_json,
    ),
    Mode.OPENROUTER_STRUCTURED_OUTPUTS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_openrouter_structured_outputs),
        reask=reask_default,
    ),
    Mode.RESPONSES_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_responses_tools),
        reask=reask_responses_tools,
    ),
    Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_responses_tools_with_inbuilt_tools),
        reask=reask_responses_tools,
    ),
    Mode.XAI_JSON: ProviderAdapter(
        prepare_request=_identity_prepare(handle_xai_json),
        reask=reask_xai_json,
    ),
    Mode.XAI_TOOLS: ProviderAdapter(
        prepare_request=_identity_prepare(handle_xai_tools),
        reask=reask_xai_tools,
    ),
}


__all__ = [
    "ProviderAdapter",
    "ProviderAdapterContext",
    "PROVIDER_ADAPTERS",
]
