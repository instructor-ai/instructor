from __future__ import annotations

import warnings

import pytest

from instructor.mode import Mode
from instructor.utils.providers import Provider

try:
    from instructor.v2.core import mode_registry, normalize_mode
except ModuleNotFoundError:
    pytest.skip("v2 module not available", allow_module_level=True)


@pytest.mark.parametrize(
    "mode",
    [
        Mode.TOOLS,
        Mode.JSON,
        Mode.JSON_SCHEMA,
        Mode.MD_JSON,
        Mode.PARALLEL_TOOLS,
        Mode.RESPONSES_TOOLS,
    ],
)
def test_normalize_mode_passthrough_for_generic_modes(mode: Mode) -> None:
    """Generic modes should pass through without warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert normalize_mode(Provider.OPENAI, mode) == mode
    assert len(caught) == 0


@pytest.mark.parametrize(
    "provider,legacy_mode",
    [
        (Provider.OPENAI, Mode.FUNCTIONS),
        (Provider.OPENAI, Mode.TOOLS_STRICT),
        (Provider.OPENAI, Mode.JSON_O1),
        (Provider.OPENAI, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS),
        (Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS),
        (Provider.ANTHROPIC, Mode.ANTHROPIC_JSON),
        (Provider.ANTHROPIC, Mode.ANTHROPIC_PARALLEL_TOOLS),
        (Provider.GENAI, Mode.GENAI_TOOLS),
        (Provider.GENAI, Mode.GENAI_JSON),
        (Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS),
        (Provider.GEMINI, Mode.GEMINI_TOOLS),
        (Provider.GEMINI, Mode.GEMINI_JSON),
        (Provider.MISTRAL, Mode.MISTRAL_TOOLS),
        (Provider.MISTRAL, Mode.MISTRAL_STRUCTURED_OUTPUTS),
        (Provider.COHERE, Mode.COHERE_TOOLS),
        (Provider.COHERE, Mode.COHERE_JSON_SCHEMA),
        (Provider.XAI, Mode.XAI_TOOLS),
        (Provider.XAI, Mode.XAI_JSON),
        (Provider.FIREWORKS, Mode.FIREWORKS_TOOLS),
        (Provider.FIREWORKS, Mode.FIREWORKS_JSON),
        (Provider.CEREBRAS, Mode.CEREBRAS_TOOLS),
        (Provider.CEREBRAS, Mode.CEREBRAS_JSON),
        (Provider.WRITER, Mode.WRITER_TOOLS),
        (Provider.WRITER, Mode.WRITER_JSON),
        (Provider.BEDROCK, Mode.BEDROCK_TOOLS),
        (Provider.BEDROCK, Mode.BEDROCK_JSON),
        (Provider.PERPLEXITY, Mode.PERPLEXITY_JSON),
        (Provider.VERTEXAI, Mode.VERTEXAI_TOOLS),
        (Provider.VERTEXAI, Mode.VERTEXAI_JSON),
        (Provider.VERTEXAI, Mode.VERTEXAI_PARALLEL_TOOLS),
        (Provider.OPENROUTER, Mode.OPENROUTER_STRUCTURED_OUTPUTS),
    ],
)
def test_legacy_modes_are_not_normalized(provider: Provider, legacy_mode: Mode) -> None:
    """Provider-specific legacy modes are rejected by the v2 registry."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert normalize_mode(provider, legacy_mode) == legacy_mode
        assert not mode_registry.is_registered(provider, legacy_mode)
    assert len(caught) == 0
