from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from types import SimpleNamespace
from typing import Any, Union

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import ResponseFunctionCallArgumentsDeltaEvent
from pydantic import BaseModel

from instructor.v2.core.errors import (
    ConfigurationError,
    IncompleteOutputException,
    ResponseParsingError,
)
from instructor.v2.core.mode import Mode
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import Partial
from instructor.v2.dsl.simple_type import ModelAdapter
from instructor.v2.providers.openai import handlers
from instructor.v2.providers.openai.handlers import (
    OpenAIHandlerBase,
    OpenAIJSONHandler,
    OpenAIJSONSchemaHandler,
    OpenAIMDJSONHandler,
    OpenAIParallelToolsHandler,
    OpenAIResponsesToolsHandler,
    OpenAIToolsHandler,
    reask_default,
    reask_responses_tools,
)


class User(BaseModel):
    name: str


class Search(BaseModel):
    query: str


def chat_completion(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    function_call: dict[str, Any] | None = None,
    refusal: str | None = None,
    finish_reason: str = "stop",
) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if function_call is not None:
        message["function_call"] = function_call
    if refusal is not None:
        message["refusal"] = refusal
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": message,
                }
            ],
        }
    )


def tool_call(name: str, arguments: str, call_id: str = "call_1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def chat_chunk(delta: dict[str, Any]) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-test",
            "choices": [{"index": 0, "finish_reason": None, "delta": delta}],
        }
    )


def base_handler(mode: Mode) -> OpenAIHandlerBase:
    handler = OpenAIToolsHandler()
    handler.mode = mode
    return handler


async def as_async(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def test_responses_tool_filter_accepts_legacy_items_and_formats_missing_details() -> (
    None
):
    legacy_call = SimpleNamespace(arguments='{"name":"Ada"}')
    ignored_message = SimpleNamespace(type="message", content="hello")

    assert handlers._filter_responses_tool_calls([ignored_message, legacy_call]) == [
        legacy_call
    ]
    assert handlers._format_responses_tool_call_details(legacy_call) == ""


def test_reask_default_keeps_assistant_message_before_correction() -> None:
    response = chat_completion(content='{"name": 2}')

    result = reask_default(
        {"messages": [{"role": "user", "content": "extract a user"}]},
        response,
        ValueError("name must be a string"),
    )

    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == '{"name": 2}'
    assert result["messages"][2] == {
        "role": "user",
        "content": (
            "Recall the function correctly, fix the errors, exceptions found\n"
            "name must be a string"
        ),
    }


def test_responses_reask_uses_legacy_arguments_without_inventing_call_details() -> None:
    response = SimpleNamespace(output=[SimpleNamespace(arguments='{"name": 3}')])

    result = reask_responses_tools(
        {"messages": []}, response, ValueError("name must be a string")
    )

    assert result["messages"] == [
        {
            "role": "user",
            "content": (
                "Validation Error found:\nname must be a string\n"
                'Recall the function correctly, fix the errors with {"name": 3}'
            ),
        }
    ]


def test_streaming_flags_ignore_none_non_classes_and_non_streaming_models() -> None:
    handler = OpenAIToolsHandler()
    iterable_user = IterableModel(User)

    handler.mark_streaming_model(None, True)
    handler.mark_streaming_model(iterable_user, False)
    handler.mark_streaming_model(User, True)

    assert handler._consume_streaming_flag(None) is False
    assert handler._consume_streaming_flag(ParallelBase(User)) is False
    assert handler._consume_streaming_flag(iterable_user) is False


@pytest.mark.parametrize(
    ("mode", "valid_chunk", "expected"),
    [
        (
            Mode.FUNCTIONS,
            chat_chunk(
                {"function_call": {"name": "User", "arguments": '{"name":"Ada"}'}}
            ),
            ['{"name":"Ada"}'],
        ),
        (Mode.JSON, chat_chunk({"content": '{"name":"Ada"}'}), ['{"name":"Ada"}']),
        (
            Mode.JSON_SCHEMA,
            chat_chunk({"content": '{"name":"Ada"}'}),
            ['{"name":"Ada"}'],
        ),
        (
            Mode.TOOLS,
            chat_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "User", "arguments": '{"name":"Ada"}'},
                        }
                    ]
                }
            ),
            ['{"name":"Ada"}'],
        ),
        (
            Mode.MD_JSON,
            chat_chunk(
                {"content": 'Here is the result:\n```json\n{"name":"Ada"}\n```'}
            ),
            ['{"name":"Ada"}\n'],
        ),
    ],
)
def test_sync_stream_extractor_handles_chat_delta_modes_and_bad_chunks(
    mode: Mode,
    valid_chunk: ChatCompletionChunk,
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Mode, "warn_mode_functions_deprecation", staticmethod(lambda: None)
    )
    chunks = [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace())]),
        valid_chunk,
    ]

    result = list(base_handler(mode).extract_streaming_json(chunks))
    if mode is Mode.MD_JSON:
        assert json.loads("".join(result)) == {"name": "Ada"}
    else:
        assert result == expected


def test_sync_responses_stream_extractor_only_yields_argument_delta_events() -> None:
    event = ResponseFunctionCallArgumentsDeltaEvent(
        delta='{"name":"Ada"}',
        item_id="fc_1",
        output_index=0,
        sequence_number=1,
        type="response.function_call_arguments.delta",
    )

    assert list(
        base_handler(Mode.RESPONSES_TOOLS).extract_streaming_json(
            [SimpleNamespace(type="response.output_text.delta"), event]
        )
    ) == ['{"name":"Ada"}']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "valid_chunk", "expected"),
    [
        (
            Mode.FUNCTIONS,
            chat_chunk(
                {"function_call": {"name": "User", "arguments": '{"name":"Ada"}'}}
            ),
            ['{"name":"Ada"}'],
        ),
        (Mode.JSON, chat_chunk({"content": '{"name":"Ada"}'}), ['{"name":"Ada"}']),
        (
            Mode.TOOLS,
            chat_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "User", "arguments": '{"name":"Ada"}'},
                        }
                    ]
                }
            ),
            ['{"name":"Ada"}'],
        ),
        (
            Mode.MD_JSON,
            chat_chunk({"content": '```json\n{"name":"Ada"}\n```'}),
            ['{"name":"Ada"}\n'],
        ),
    ],
)
async def test_async_stream_extractor_handles_chat_delta_modes_and_bad_chunks(
    mode: Mode,
    valid_chunk: ChatCompletionChunk,
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Mode, "warn_mode_functions_deprecation", staticmethod(lambda: None)
    )
    chunks = [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace())]),
        valid_chunk,
    ]

    result = [
        part
        async for part in base_handler(mode).extract_streaming_json_async(
            as_async(chunks)
        )
    ]
    if mode is Mode.MD_JSON:
        assert json.loads("".join(result)) == {"name": "Ada"}
    else:
        assert result == expected


@pytest.mark.asyncio
async def test_async_responses_stream_extractor_only_yields_argument_delta_events() -> (
    None
):
    event = ResponseFunctionCallArgumentsDeltaEvent(
        delta='{"name":"Ada"}',
        item_id="fc_1",
        output_index=0,
        sequence_number=1,
        type="response.function_call_arguments.delta",
    )

    assert [
        part
        async for part in base_handler(
            Mode.RESPONSES_TOOLS
        ).extract_streaming_json_async(
            as_async([SimpleNamespace(type="response.output_text.delta"), event])
        )
    ] == ['{"name":"Ada"}']


def test_parse_streaming_response_forwards_context_strict_and_mode() -> None:
    received: dict[str, Any] = {}

    class StreamingUser(BaseModel):
        @classmethod
        def from_streaming_response(
            cls,
            response: list[Any],
            *,
            stream_extractor: Any,
            mode: Mode,
            **kwargs: Any,
        ) -> Iterable[User]:
            received.update(
                response=response, extractor=stream_extractor, mode=mode, **kwargs
            )
            return iter([User(name="Ada")])

    response = [chat_chunk({"content": '{"name":"Ada"}'})]
    result = base_handler(Mode.JSON)._parse_streaming_response(
        StreamingUser,
        response,
        validation_context={"source": "chat"},
        strict=True,
    )

    assert result == [User(name="Ada")]
    assert received["response"] is response
    assert received["mode"] is Mode.JSON
    assert received["context"] == {"source": "chat"}
    assert received["strict"] is True
    assert callable(received["extractor"])


def test_parse_streaming_response_falls_back_when_signature_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StreamingUser(BaseModel):
        @classmethod
        def from_streaming_response(
            cls, _response: list[Any], *, stream_extractor: Any, **kwargs: Any
        ) -> Iterable[User]:
            assert "mode" not in kwargs
            assert callable(stream_extractor)
            return iter([User(name="Ada")])

    original_signature = handlers.inspect.signature

    def unavailable_signature(value: Any) -> Any:
        if getattr(value, "__name__", "") == "from_streaming_response":
            raise ValueError("extension method has no signature")
        return original_signature(value)

    monkeypatch.setattr(handlers.inspect, "signature", unavailable_signature)

    assert base_handler(Mode.JSON)._parse_streaming_response(
        StreamingUser, [], validation_context=None, strict=None
    ) == [User(name="Ada")]


@pytest.mark.asyncio
async def test_parse_streaming_response_returns_async_iterable_results() -> None:
    iterable_user = IterableModel(User)
    chunks = as_async([chat_chunk({"content": '{"tasks":[{"name":"Ada"}]}'})])

    result = base_handler(Mode.JSON)._parse_streaming_response(
        iterable_user, chunks, validation_context=None, strict=None
    )

    assert [item async for item in result] == [User(name="Ada")]


def test_finalize_parsed_result_handles_parallel_iterable_adapter_and_base_model() -> (
    None
):
    handler = OpenAIToolsHandler()
    response = chat_completion(content="unused")
    iterable_user = IterableModel(User)
    adapter = ModelAdapter[str]
    parallel = ParallelBase(User, Search)

    assert handler._finalize_parsed_result(
        iterable_user,
        response,
        iterable_user(tasks=[User(name="Ada")]),
    ) == [User(name="Ada")]
    assert handler._finalize_parsed_result(parallel, response, ("kept",)) == ("kept",)
    assert (
        handler._finalize_parsed_result(adapter, response, adapter(content="ok"))
        == "ok"
    )
    parsed_user = User(name="Ada")
    assert handler._finalize_parsed_result(User, response, parsed_user) is parsed_user
    assert parsed_user._raw_response is response


def test_extract_tool_call_json_supports_legacy_and_serializable_arguments() -> None:
    handler = OpenAIToolsHandler()
    legacy = chat_completion(
        function_call={"name": "User", "arguments": '{"name":"Ada"}'}
    )
    dict_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    refusal=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(arguments={"name": "Ada"})
                        )
                    ],
                )
            )
        ]
    )
    list_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    refusal=None,
                    tool_calls=[
                        SimpleNamespace(function=SimpleNamespace(arguments=["Ada"]))
                    ],
                )
            )
        ]
    )

    assert handler._extract_tool_call_json(legacy) == '{"name":"Ada"}'
    assert json.loads(handler._extract_tool_call_json(dict_response)) == {"name": "Ada"}
    assert json.loads(handler._extract_tool_call_json(list_response)) == ["Ada"]


@pytest.mark.parametrize(
    ("response", "error", "message"),
    [
        (
            chat_completion(refusal="policy"),
            AssertionError,
            "Unable to generate a response due to policy",
        ),
        (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            refusal=None,
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(arguments=None)
                                )
                            ],
                        )
                    )
                ]
            ),
            ResponseParsingError,
            "Tool call arguments missing in response",
        ),
        (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            refusal=None,
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(arguments=object())
                                )
                            ],
                        )
                    )
                ]
            ),
            ResponseParsingError,
            "Tool call arguments must be JSON-serializable",
        ),
        (
            chat_completion(content="no tool call", tool_calls=[]),
            ResponseParsingError,
            "No tool calls or function call found in response",
        ),
    ],
)
def test_extract_tool_call_json_reports_refusals_and_malformed_responses(
    response: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        OpenAIToolsHandler()._extract_tool_call_json(response)


def test_tools_prepare_and_parse_parallel_calls_and_ignore_unknown_tool() -> None:
    handler = OpenAIToolsHandler()
    response_model = Iterable[Union[User, Search]]

    prepared_model, kwargs = handler.prepare_request(response_model, {"messages": []})
    response = chat_completion(
        tool_calls=[
            tool_call("User", '{"name":"Ada"}', "call_user"),
            tool_call("Search", '{"query":"python"}', "call_search"),
            tool_call("Unrelated", "{}", "call_unknown"),
        ]
    )

    assert prepared_model is response_model
    assert kwargs["tool_choice"] == "auto"
    assert {tool["function"]["name"] for tool in kwargs["tools"]} == {"User", "Search"}
    assert list(
        handler.parse_response(
            response,
            prepared_model,
            validation_context={"source": "parallel"},
            strict=True,
        )
    ) == [User(name="Ada"), Search(query="python")]


def test_tools_prepare_strict_schema_and_parse_incomplete_output() -> None:
    handler = OpenAIToolsHandler()

    prepared_model, kwargs = handler.prepare_request(User, {"strict": True})

    assert prepared_model.__name__ == "User"
    assert prepared_model.model_fields["name"].annotation is str
    assert kwargs["tools"][0]["function"]["strict"] is True
    assert "strict" not in kwargs
    with pytest.raises(IncompleteOutputException):
        handler.parse_response(
            chat_completion(
                tool_calls=[tool_call("User", '{"name":"Ada"}')], finish_reason="length"
            ),
            User,
        )


@pytest.mark.parametrize(
    "handler",
    [OpenAIJSONSchemaHandler(), OpenAIJSONHandler(), OpenAIMDJSONHandler()],
)
def test_json_handlers_reject_incomplete_output(handler: OpenAIHandlerBase) -> None:
    with pytest.raises(IncompleteOutputException):
        handler.parse_response(
            chat_completion(content="{", finish_reason="length"), User
        )


def test_json_schema_stream_parses_iterable_and_partial_models() -> None:
    handler = OpenAIJSONSchemaHandler()
    iterable_user = IterableModel(User)
    partial_user = Partial[User]

    iterable_result = handler.parse_response(
        [chat_chunk({"content": '{"tasks":[{"name":"Ada"}]}'})],
        iterable_user,
        stream=True,
    )
    partial_result = handler.parse_response(
        [chat_chunk({"content": '{"name":"Ada"}'})], partial_user, stream=True
    )

    assert list(iterable_result) == [User(name="Ada")]
    assert partial_result[-1] == User(name="Ada")


@pytest.mark.parametrize("handler", [OpenAIJSONHandler(), OpenAIMDJSONHandler()])
def test_json_prompt_handlers_extend_list_system_content_and_create_system_message(
    handler: OpenAIHandlerBase,
) -> None:
    list_messages = [
        {"role": "system", "content": [{"type": "text", "text": "Extract users."}]}
    ]

    _, with_list = handler.prepare_request(User, {"messages": list_messages})
    _, without_messages = handler.prepare_request(User, {"messages": []})

    assert "Extract users." in with_list["messages"][0]["content"][0]["text"]
    assert "json_schema" in with_list["messages"][0]["content"][0]["text"]
    assert without_messages["messages"][0]["role"] == "system"
    assert "User" in without_messages["messages"][0]["content"]
    if isinstance(handler, OpenAIMDJSONHandler):
        assert "```json codeblock" in with_list["messages"][-1]["content"][0]["text"]


@pytest.mark.parametrize(
    ("handler", "content"),
    [
        (OpenAIJSONHandler(), '{"tasks":[{"name":"Ada"}]}'),
        (OpenAIMDJSONHandler(), '```json\n{"tasks":[{"name":"Ada"}]}\n```'),
    ],
)
def test_json_prompt_handlers_consume_registered_streaming_model(
    handler: OpenAIHandlerBase, content: str
) -> None:
    iterable_user = IterableModel(User)
    _, kwargs = handler.prepare_request(iterable_user, {"stream": True})

    parsed = handler.parse_response([chat_chunk({"content": content})], iterable_user)

    assert kwargs["stream"] is True
    assert list(parsed) == [User(name="Ada")]
    assert handler._consume_streaming_flag(iterable_user) is False


def test_parallel_tools_prepare_handles_none_streaming_and_model_union() -> None:
    handler = OpenAIParallelToolsHandler()
    response_model = Iterable[Union[User, Search]]

    assert handler.prepare_request(None, {"messages": []}) == (None, {"messages": []})
    with pytest.raises(ConfigurationError, match="stream=True is not supported"):
        handler.prepare_request(response_model, {"stream": True})
    prepared, kwargs = handler.prepare_request(response_model, {"messages": []})

    assert isinstance(prepared, ParallelBase)
    assert prepared.registry == {"User": User, "Search": Search}
    assert kwargs["tool_choice"] == "auto"
    assert {tool["function"]["name"] for tool in kwargs["tools"]} == {"User", "Search"}


def test_parallel_tools_parse_valid_calls_and_report_empty_or_incomplete_output() -> (
    None
):
    handler = OpenAIParallelToolsHandler()
    response_model = Iterable[Union[User, Search]]
    valid = chat_completion(
        tool_calls=[
            tool_call("User", '{"name":"Ada"}', "call_user"),
            tool_call("Search", '{"query":"python"}', "call_search"),
            tool_call("Unrelated", "{}", "call_unknown"),
        ]
    )

    assert list(
        handler.parse_response(
            valid,
            response_model,
            validation_context={"source": "parallel"},
            strict=True,
        )
    ) == [User(name="Ada"), Search(query="python")]
    with pytest.raises(ResponseParsingError, match="No tool calls in response"):
        handler.parse_response(chat_completion(tool_calls=[]), response_model)
    with pytest.raises(IncompleteOutputException):
        handler.parse_response(
            chat_completion(tool_calls=[], finish_reason="length"), response_model
        )


def test_responses_tools_converts_max_tokens_and_falls_back_to_chat_tool_call() -> None:
    handler = OpenAIResponsesToolsHandler()

    prepared, kwargs = handler.prepare_request(User, {"max_tokens": 64})
    parsed = handler.parse_response(
        chat_completion(tool_calls=[tool_call("User", '{"name":"Ada"}')]),
        prepared,
        validation_context={"source": "fallback"},
        strict=True,
    )

    assert prepared.__name__ == "User"
    assert prepared.model_fields["name"].annotation is str
    assert kwargs["max_output_tokens"] == 64
    assert "max_tokens" not in kwargs
    assert parsed.model_dump() == {"name": "Ada"}
    assert parsed._raw_response.choices[0].message.tool_calls[0].function.name == "User"
