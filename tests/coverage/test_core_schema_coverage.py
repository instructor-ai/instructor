"""Coverage tests for v2 response-schema and response-model edge cases."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, ForwardRef

import pytest
from anthropic.types import Message
from openai.types.chat import ChatCompletion
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator
from typing_extensions import TypedDict

from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.exceptions import (
    RegistryError,
    RegistryValidationMixin,
    ValidationContextError,
)
from instructor.v2.core.function_calls import (
    ResponseSchema,
    _extract_text_content,
    _handle_incomplete_output,
    _validate_model_from_json,
)
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core import response_model as response_model_module
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.core.schema import generate_gemini_schema
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.providers.openai.handlers import OpenAIToolsHandler


class Answer(ResponseSchema):
    value: int

    @field_validator("value")
    @classmethod
    def validate_context(cls, value: int, info: ValidationInfo) -> int:
        if info.context is not None:
            assert info.context["source"] == "coverage"
        return value


class Row(TypedDict):
    value: int


def _chat_completion(
    *,
    content: str | None = None,
    function_arguments: str | None = None,
    tool_arguments: str | None = None,
    finish_reason: str = "stop",
) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if function_arguments is not None:
        message["function_call"] = {
            "name": "Answer",
            "arguments": function_arguments,
        }
    if tool_arguments is not None:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "Answer", "arguments": tool_arguments},
            }
        ]
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
        }
    )


def _anthropic_message(*, text: str | None = None, tool: bool = False) -> Message:
    content: list[dict[str, Any]] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    if tool:
        content.append(
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "Answer",
                "input": {"value": 7},
            }
        )
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": content,
            "stop_reason": "tool_use" if tool else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def test_incomplete_openai_and_anthropic_outputs_keep_last_completion() -> None:
    openai_completion = _chat_completion(content="partial", finish_reason="length")
    with pytest.raises(IncompleteOutputException) as openai_error:
        _handle_incomplete_output(openai_completion)
    assert openai_error.value.last_completion is openai_completion

    anthropic_completion = _anthropic_message(text="partial").model_copy(
        update={"stop_reason": "max_tokens"}
    )
    with pytest.raises(IncompleteOutputException) as anthropic_error:
        _handle_incomplete_output(anthropic_completion)
    assert anthropic_error.value.last_completion is anthropic_completion


@pytest.mark.parametrize(
    "completion",
    [
        {"output": None},
        {"output": {"message": None}},
        {"output": {"message": {"content": "not-a-list"}}},
        {"output": {"message": {"content": []}}},
        {"output": {"message": {"content": [None]}}},
    ],
)
def test_extract_text_content_tolerates_malformed_bedrock_responses(
    completion: dict[str, Any],
) -> None:
    assert _extract_text_content(completion) == ""


def test_validate_non_model_type_uses_type_adapter_and_honors_strictness() -> None:
    assert _validate_model_from_json(list[int], '["1", 2]', strict=False) == [1, 2]
    assert _validate_model_from_json(list[int], "[1, 2]", strict=True) == [1, 2]

    with pytest.raises(ValidationError, match="valid integer"):
        _validate_model_from_json(list[int], '["1", 2]', strict=True)


@pytest.mark.parametrize(
    ("method_name", "completion"),
    [
        ("parse_anthropic_tools", _anthropic_message(tool=True)),
        ("parse_anthropic_json", _anthropic_message(text='{"value": 7}')),
        (
            "parse_functions",
            _chat_completion(function_arguments='{"value": 7}'),
        ),
        (
            "parse_responses_tools",
            SimpleNamespace(
                output=[
                    ResponseFunctionToolCall(
                        type="function_call",
                        id="fc_1",
                        call_id="call_1",
                        name="Answer",
                        arguments='{"value": 7}',
                    )
                ]
            ),
        ),
        ("parse_tools", _chat_completion(tool_arguments='{"value": 7}')),
        ("parse_json", _chat_completion(content='{"value": 7}')),
    ],
)
def test_legacy_parsers_accept_real_provider_response_shapes(
    method_name: str, completion: Any
) -> None:
    with pytest.warns(DeprecationWarning) as warnings:
        parsed = getattr(Answer, method_name)(
            completion,
            validation_context={"source": "coverage"},
            strict=True,
        )

    assert any(f"{method_name} is deprecated" in str(item.message) for item in warnings)
    assert isinstance(parsed, Answer)
    assert parsed.value == 7


def test_prepare_response_model_keeps_model_lists_and_converts_typed_dicts() -> None:
    prepared_models = prepare_response_model(list[Answer])
    prepared_rows = prepare_response_model(Iterable[Row])

    assert prepared_models is not None
    assert issubclass(prepared_models, IterableBase)
    assert prepared_models.model_fields["tasks"].annotation == list[Answer]
    assert prepared_rows is not None
    assert issubclass(prepared_rows, IterableBase)
    row_type = prepared_rows.model_fields["tasks"].annotation.__args__[0]
    assert issubclass(row_type, BaseModel)
    assert row_type.model_validate({"value": "3"}).value == 3


def test_prepare_response_model_preserves_model_list_when_type_is_broadly_classified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_list = list[Answer]
    original = response_model_module.is_simple_type
    monkeypatch.setattr(
        response_model_module,
        "is_simple_type",
        lambda typehint: typehint == model_list or original(typehint),
    )

    prepared = prepare_response_model(model_list)

    assert prepared is not None
    assert issubclass(prepared, IterableBase)
    assert prepared.model_fields["tasks"].annotation == list[Answer]


def test_prepare_response_model_rejects_unparameterized_and_forward_types() -> None:
    with pytest.raises(ValueError, match="must be parameterized"):
        prepare_response_model(Iterable[()])

    with pytest.raises(TypeError, match="subclass of pydantic.BaseModel"):
        prepare_response_model(ForwardRef("MissingModel"))


def test_gemini_schema_compatibility_wrapper_is_explicit_about_missing_sdk() -> None:
    with pytest.warns(DeprecationWarning, match="generate_gemini_schema is deprecated"):
        try:
            schema = generate_gemini_schema(Answer)
        except ImportError as error:
            assert "Please install google-genai instead" in str(error)
        else:
            assert schema.name == "Answer"


def test_mode_handler_repr_names_the_concrete_handler() -> None:
    assert repr(OpenAIToolsHandler()) == "<OpenAIToolsHandler>"


def test_registry_validation_reports_provider_mode_and_available_modes() -> None:
    with pytest.raises(RegistryError) as error:
        RegistryValidationMixin.validate_mode_registration(
            Provider.ANTHROPIC, Mode.RESPONSES_TOOLS
        )

    message = str(error.value)
    assert "RESPONSES_TOOLS" in message
    assert "ANTHROPIC" in message
    assert "Available modes:" in message
    assert "OPENAI" in message


def test_validation_context_prefers_context_and_warns_for_the_old_name() -> None:
    assert RegistryValidationMixin.validate_context_parameters(
        {"source": "new"}, None
    ) == {"source": "new"}

    with pytest.warns(DeprecationWarning, match="'validation_context' is deprecated"):
        deprecated = RegistryValidationMixin.validate_context_parameters(
            None, {"source": "old"}
        )
    assert deprecated == {"source": "old"}

    with pytest.raises(ValidationContextError, match="Cannot provide both"):
        RegistryValidationMixin.validate_context_parameters(
            {"source": "new"}, {"source": "old"}
        )
