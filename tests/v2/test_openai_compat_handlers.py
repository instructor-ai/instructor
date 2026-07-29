"""Shared handler tests for OpenAI-compatible v2 providers."""

from __future__ import annotations

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.core.exceptions import IncompleteOutputException
from instructor.v2.core.registry import mode_registry, normalize_mode
from tests.coverage._openai import chat_completion, tool_call
from tests.v2.provider_matrix import PROVIDER_SPECS

OPENAI_COMPAT_PROVIDERS = (
    Provider.GROQ,
    Provider.FIREWORKS,
    Provider.CEREBRAS,
    Provider.WRITER,
)
OPENAI_HANDLER_ALIAS_PROVIDERS = (
    Provider.GROQ,
    Provider.FIREWORKS,
    Provider.CEREBRAS,
)


class Answer(BaseModel):
    answer: float


def _handlers(provider: Provider, mode: Mode):
    return mode_registry.get_handlers(provider, mode)


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
def test_tools_request_preserves_kwargs(provider: Provider) -> None:
    kwargs = {
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 500,
        "temperature": 0.7,
    }
    original = kwargs.copy()

    _, result_kwargs = _handlers(provider, Mode.TOOLS).request_handler(Answer, kwargs)

    assert kwargs == original
    assert result_kwargs["max_tokens"] == 500
    assert result_kwargs["temperature"] == 0.7
    assert result_kwargs["tools"][0]["type"] == "function"


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
def test_md_json_extends_existing_system_message(provider: Provider) -> None:
    kwargs = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
    }

    _, result_kwargs = _handlers(provider, Mode.MD_JSON).request_handler(Answer, kwargs)

    system_message = result_kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert "You are helpful." in system_message["content"]
    assert "json_schema" in system_message["content"]


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
@pytest.mark.parametrize(
    ("mode", "response"),
    [
        (
            Mode.TOOLS,
            chat_completion(tool_calls=[tool_call("Answer", {"answer": 5.0})]),
        ),
        (Mode.MD_JSON, chat_completion(content='{"answer": 21.0}')),
    ],
)
def test_response_parser_common_paths(
    provider: Provider,
    mode: Mode,
    response: ChatCompletion,
) -> None:
    result = _handlers(provider, mode).response_parser(
        response,
        Answer,
        validation_context={"test": "context"},
        strict=True,
    )

    assert isinstance(result, Answer)


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
def test_tools_support_nested_models(provider: Provider) -> None:
    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    _, result_kwargs = _handlers(provider, Mode.TOOLS).request_handler(
        Person,
        {"messages": [{"role": "user", "content": "Get person info"}]},
    )

    assert "tools" in result_kwargs


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
def test_incomplete_tools_output_raises(provider: Provider) -> None:
    response = chat_completion(
        tool_calls=[tool_call("Answer", {"answer": 4.0})],
        finish_reason="length",
    )

    with pytest.raises(IncompleteOutputException):
        _handlers(provider, Mode.TOOLS).response_parser(response, Answer)


@pytest.mark.parametrize("provider", OPENAI_HANDLER_ALIAS_PROVIDERS)
@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.MD_JSON])
def test_alias_providers_reuse_openai_handlers(provider: Provider, mode: Mode) -> None:
    provider_handlers = _handlers(provider, mode)
    openai_handlers = _handlers(Provider.OPENAI, mode)

    assert provider_handlers.request_handler == openai_handlers.request_handler
    assert provider_handlers.response_parser == openai_handlers.response_parser


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
def test_legacy_modes_remain_accepted(provider: Provider) -> None:
    spec = PROVIDER_SPECS[provider]

    for legacy_mode in spec.legacy_modes:
        assert normalize_mode(provider, legacy_mode) != legacy_mode
        assert mode_registry.is_registered(provider, legacy_mode)
