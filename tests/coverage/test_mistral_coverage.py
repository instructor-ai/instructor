from __future__ import annotations

import builtins
import runpy
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from mistralai import Mistral
from mistralai.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    CompletionChunk,
    CompletionEvent,
    CompletionResponseStreamChoice,
    DeltaMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from pydantic import BaseModel

import instructor.v2.providers.mistral.client as mistral_client
from instructor.v2.core.errors import ClientError, ModeError
from instructor.v2.core.mode import Mode
from instructor.v2.core.multimodal import PDF
from instructor.v2.core.providers import Provider
from instructor.v2.dsl.iterable import IterableBase, IterableModel
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import Partial
from instructor.v2.dsl.simple_type import ModelAdapter
from instructor.v2.providers.mistral.handlers import (
    MistralJSONSchemaHandler,
    MistralMDJSONHandler,
    MistralToolsHandler,
)
from instructor.v2.providers.mistral.multimodal import pdf_to_mistral
from tests.coverage._streams import async_items


class User(BaseModel):
    name: str
    age: int


class Answer(BaseModel):
    answer: float


def tool_call(name: str, arguments: dict[str, Any] | str, call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments=arguments),
    )


def response(
    *,
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="mistral-response",
        object="chat.completion",
        model="mistral-small-latest",
        created=1,
        usage=UsageInfo(prompt_tokens=8, completion_tokens=5, total_tokens=13),
        choices=[
            ChatCompletionChoice(
                index=0,
                finish_reason=finish_reason,
                message=AssistantMessage(content=content, tool_calls=tool_calls),
            )
        ],
    )


def event(
    *, content: str | None = None, tool_calls: list[ToolCall] | None = None
) -> CompletionEvent:
    return CompletionEvent(
        data=CompletionChunk(
            id="mistral-stream",
            model="mistral-small-latest",
            choices=[
                CompletionResponseStreamChoice(
                    index=0,
                    finish_reason=None,
                    delta=DeltaMessage(content=content, tool_calls=tool_calls),
                )
            ],
        )
    )


class FakeMistral:
    def __init__(self) -> None:
        self.chat = MagicMock()
        self.chat.complete_async = AsyncMock()
        self.chat.stream_async = AsyncMock()


def test_missing_mistral_sdk_has_clear_client_and_package_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__
    client_path = Path(mistral_client.__file__)
    package_path = client_path.with_name("__init__.py")

    def without_sdk(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "mistralai":
            raise ImportError("mistralai is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_sdk)
    unloaded_client = runpy.run_path(str(client_path))

    assert unloaded_client["Mistral"] is None
    with pytest.raises(ClientError, match="pip install mistralai"):
        unloaded_client["from_mistral"](None)

    def without_client(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "instructor.v2.providers.mistral.client":
            raise ImportError("client dependencies are unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_client)
    unloaded_package = runpy.run_path(str(package_path))

    assert unloaded_package["from_mistral"] is None
    assert unloaded_package["__all__"] == ["from_mistral"]


def test_from_mistral_rejects_bad_mode_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mistral_client, "Mistral", FakeMistral)

    with pytest.raises(ModeError) as mode_error:
        mistral_client.from_mistral(
            cast(Mistral, FakeMistral()), mode=Mode.ANTHROPIC_JSON
        )
    assert "anthropic_json" in str(mode_error.value)
    assert Provider.MISTRAL.value in str(mode_error.value)

    with pytest.raises(ClientError, match="Got: object"):
        mistral_client.from_mistral(cast(Mistral, object()), mode=Mode.TOOLS)


def test_from_mistral_warns_for_deprecated_tools_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mistral_client, "Mistral", FakeMistral)

    with pytest.warns(DeprecationWarning, match="Use Mode.TOOLS instead"):
        client = mistral_client.from_mistral(
            cast(Mistral, FakeMistral()), mode=Mode.MISTRAL_TOOLS
        )

    assert client.mode is Mode.TOOLS


def test_sync_client_routes_completion_retry_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mistral_client, "Mistral", FakeMistral)
    sdk = FakeMistral()
    valid_response = response(
        tool_calls=[tool_call("User", {"name": "Ada", "age": 36}, "second")]
    )
    sdk.chat.complete.side_effect = [
        response(
            tool_calls=[tool_call("User", {"name": "Ada", "age": "bad"}, "first")]
        ),
        valid_response,
    ]
    sdk.chat.stream.return_value = [
        event(tool_calls=[]),
        event(tool_calls=[tool_call("PartialUser", '{"name":"Ada"', "partial-1")]),
        event(tool_calls=[tool_call("PartialUser", ',"age":36}', "partial-2")]),
    ]

    client = mistral_client.from_mistral(
        cast(Mistral, sdk),
        mode=Mode.TOOLS,
        model="mistral-small-latest",
        temperature=0.2,
    )
    result = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract Ada"}],
        max_retries=2,
    )
    streamed = list(
        client.create_partial(
            response_model=User,
            messages=[{"role": "user", "content": "Stream Ada"}],
        )
    )

    assert isinstance(result, User)
    assert result.model_dump() == {"name": "Ada", "age": 36}
    assert cast(Any, result)._raw_response is valid_response
    assert sdk.chat.complete.call_count == 2
    retry_kwargs = sdk.chat.complete.call_args_list[1].kwargs
    assert retry_kwargs["model"] == "mistral-small-latest"
    assert retry_kwargs["tool_choice"] == "any"
    assert retry_kwargs["temperature"] == 0.2
    assert retry_kwargs["messages"][-2]["role"] == "assistant"
    assert retry_kwargs["messages"][-1]["role"] == "tool"
    assert retry_kwargs["messages"][-1]["tool_call_id"] == "first"
    assert "Validation Error found" in retry_kwargs["messages"][-1]["content"]
    assert sdk.chat.stream.call_count == 1
    assert "stream" not in sdk.chat.stream.call_args.kwargs
    assert streamed[-1].name == "Ada"
    assert streamed[-1].age == 36


@pytest.mark.asyncio
async def test_async_client_routes_completion_retry_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mistral_client, "Mistral", FakeMistral)
    sdk = FakeMistral()
    valid_response = response(
        tool_calls=[tool_call("User", {"name": "Grace", "age": 40}, "second")]
    )
    sdk.chat.complete_async.side_effect = [
        response(
            tool_calls=[tool_call("User", {"name": "Grace", "age": "bad"}, "first")]
        ),
        valid_response,
    ]
    sdk.chat.stream_async.return_value = async_items(
        [
            event(tool_calls=[]),
            event(
                tool_calls=[tool_call("PartialUser", '{"name":"Grace"', "partial-1")]
            ),
            event(tool_calls=[tool_call("PartialUser", ',"age":40}', "partial-2")]),
        ]
    )

    client = mistral_client.from_mistral(
        cast(Mistral, sdk),
        mode=Mode.TOOLS,
        use_async=True,
        model="mistral-small-latest",
        temperature=0.1,
    )
    result = await client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract Grace"}],
        max_retries=2,
    )
    streamed = [
        item
        async for item in client.create_partial(
            response_model=User,
            messages=[{"role": "user", "content": "Stream Grace"}],
        )
    ]

    assert isinstance(result, User)
    assert result.model_dump() == {"name": "Grace", "age": 40}
    assert cast(Any, result)._raw_response is valid_response
    assert sdk.chat.complete_async.await_count == 2
    retry_kwargs = sdk.chat.complete_async.await_args_list[1].kwargs
    assert retry_kwargs["model"] == "mistral-small-latest"
    assert retry_kwargs["tool_choice"] == "any"
    assert retry_kwargs["messages"][-1]["tool_call_id"] == "first"
    assert "Validation Error found" in retry_kwargs["messages"][-1]["content"]
    sdk.chat.stream_async.assert_awaited_once()
    assert "stream" not in sdk.chat.stream_async.await_args.kwargs
    assert streamed[-1].name == "Grace"
    assert streamed[-1].age == 40


def test_streaming_flags_and_sync_extractors_skip_empty_and_bad_chunks() -> None:
    tools = MistralToolsHandler()
    schema = MistralJSONSchemaHandler()
    markdown = MistralMDJSONHandler()
    users = IterableModel(User)

    tools.mark_streaming_model(None, True)
    tools.mark_streaming_model(users, False)
    tools.mark_streaming_model(User, True)
    assert tools._consume_streaming_flag(None) is False
    assert tools._consume_streaming_flag(cast(type[BaseModel], object())) is False
    assert tools._consume_streaming_flag(User) is False
    tools.mark_streaming_model(users, True)
    assert tools._consume_streaming_flag(users) is True
    assert tools._consume_streaming_flag(users) is False

    tool_chunks = [
        object(),
        event(tool_calls=[]),
        event(tool_calls=[tool_call("Users", '{"tasks":[]}', "chunk")]),
    ]
    assert list(tools.extract_streaming_json(tool_chunks)) == ['{"tasks":[]}']
    assert list(
        schema.extract_streaming_json([object(), event(content='{"answer":2}')])
    ) == ['{"answer":2}']
    assert (
        "".join(
            markdown.extract_streaming_json(
                [
                    object(),
                    event(content="```json\n"),
                    event(content='{"answer": 3}'),
                    event(content="\n```"),
                ]
            )
        )
        == '{"answer": 3}'
    )


@pytest.mark.asyncio
async def test_async_extractors_skip_empty_and_bad_chunks() -> None:
    tools = MistralToolsHandler()
    schema = MistralJSONSchemaHandler()
    markdown = MistralMDJSONHandler()

    tool_chunks = [
        object(),
        event(tool_calls=[]),
        event(tool_calls=[tool_call("Users", '{"tasks":[]}', "chunk")]),
    ]
    assert [
        chunk
        async for chunk in tools.extract_streaming_json_async(async_items(tool_chunks))
    ] == ['{"tasks":[]}']
    assert [
        chunk
        async for chunk in schema.extract_streaming_json_async(
            async_items([object(), event(content='{"answer":2}')])
        )
    ] == ['{"answer":2}']
    assert (
        "".join(
            [
                chunk
                async for chunk in markdown.extract_streaming_json_async(
                    async_items(
                        [
                            object(),
                            event(content="```json\n"),
                            event(content='{"answer": 3}'),
                            event(content="\n```"),
                        ]
                    )
                )
            ]
        )
        == '{"answer": 3}'
    )


def test_parallel_tool_request_and_response_support_multiple_models() -> None:
    handler = MistralToolsHandler()
    model = Iterable[Union[User, Answer]]
    original = {"messages": [{"role": "user", "content": "Extract all results"}]}

    returned_model, request = handler.prepare_request(
        cast(type[BaseModel], model), original
    )
    assert returned_model is not None
    parsed = list(
        handler.parse_response(
            response(
                tool_calls=[
                    tool_call("User", {"name": "Ada", "age": 36}, "user"),
                    tool_call("Unknown", {"ignored": True}, "unknown"),
                    tool_call("Answer", '{"answer": 42.0}', "answer"),
                ]
            ),
            returned_model,
            validation_context={"request_id": "parallel"},
            strict=True,
        )
    )

    assert returned_model == model
    assert [tool["function"]["name"] for tool in request["tools"]] == ["User", "Answer"]
    assert request["tool_choice"] == "any"
    assert "tools" not in original
    assert parsed == [User(name="Ada", age=36), Answer(answer=42.0)]


def test_tools_streaming_iterable_parser_uses_task_list_chunks() -> None:
    handler = MistralToolsHandler()
    model, request = handler.prepare_request(
        cast(type[BaseModel], Iterable[User]),
        {"messages": [{"role": "user", "content": "Extract users"}], "stream": True},
    )
    stream = [
        event(tool_calls=[]),
        event(
            tool_calls=[
                tool_call(
                    "IterableUser",
                    '{"tasks":[{"name":"Ada","age":36},{"name":"Grace","age":40}]}',
                    "users",
                )
            ]
        ),
    ]

    assert model is not None
    parsed = list(
        handler.parse_response(
            stream, model, validation_context={"request_id": "stream"}, strict=True
        )
    )

    assert request["stream"] is True
    assert len(request["tools"]) == 1
    assert issubclass(model, IterableBase)
    assert parsed == [User(name="Ada", age=36), User(name="Grace", age=40)]


@pytest.mark.asyncio
async def test_tools_streaming_iterable_parser_supports_async_streams() -> None:
    handler = MistralToolsHandler()
    model, _ = handler.prepare_request(
        cast(type[BaseModel], Iterable[User]),
        {"messages": [{"role": "user", "content": "Extract users"}], "stream": True},
    )
    stream = async_items(
        [
            event(tool_calls=[]),
            event(
                tool_calls=[
                    tool_call(
                        "IterableUser",
                        '{"tasks":[{"name":"Ada","age":36},{"name":"Grace","age":40}]}',
                        "users",
                    )
                ]
            ),
        ]
    )

    assert model is not None
    parsed = [
        item
        async for item in handler.parse_response(
            stream,
            model,
            validation_context={"request_id": "async-stream"},
            strict=True,
        )
    ]

    assert parsed == [User(name="Ada", age=36), User(name="Grace", age=40)]


@pytest.mark.asyncio
async def test_json_and_markdown_streaming_partial_models_and_empty_messages() -> None:
    schema = MistralJSONSchemaHandler()
    markdown = MistralMDJSONHandler()
    partial_user = Partial[User]
    passthrough = {"messages": [{"role": "user", "content": "No schema"}]}
    assert schema.prepare_request(None, passthrough) == (None, passthrough)
    schema_model, schema_request = schema.prepare_request(
        partial_user,
        {
            "messages": [{"role": "user", "content": "Extract Ada"}],
            "tools": [{"type": "function"}],
            "tool_choice": "any",
            "stream": True,
        },
    )
    markdown_model, markdown_request = markdown.prepare_request(
        partial_user, {"messages": [], "stream": True}
    )

    assert schema_model is not None
    assert markdown_model is not None
    schema_stream = [event(content='{"name":"Ada"'), event(content=',"age":36}')]
    markdown_stream = [
        event(content="```json\n"),
        event(content='{"name":"Grace","age":40}'),
        event(content="\n```"),
    ]
    schema_items = schema.parse_response(
        schema_stream,
        schema_model,
        validation_context={"request_id": "schema"},
        strict=True,
    )
    markdown_items = markdown.parse_response(
        markdown_stream,
        markdown_model,
        validation_context={"request_id": "markdown"},
        strict=True,
    )
    async_schema_model, _ = schema.prepare_request(
        partial_user,
        {"messages": [{"role": "user", "content": "Extract Ada"}], "stream": True},
    )
    async_markdown_model, _ = markdown.prepare_request(
        partial_user,
        {"messages": [{"role": "user", "content": "Extract Grace"}], "stream": True},
    )
    assert async_schema_model is not None
    assert async_markdown_model is not None
    async_schema_items = [
        item
        async for item in schema.parse_response(
            async_items(schema_stream),
            async_schema_model,
            validation_context={"request_id": "async-schema"},
            strict=True,
        )
    ]
    async_markdown_items = [
        item
        async for item in markdown.parse_response(
            async_items(markdown_stream),
            async_markdown_model,
            validation_context={"request_id": "async-markdown"},
        )
    ]

    assert "response_format" in schema_request
    assert "tools" not in schema_request
    assert "tool_choice" not in schema_request
    assert schema_items[-1].name == "Ada"
    assert schema_items[-1].age == 36
    assert async_schema_items[-1].model_dump() == {"name": "Ada", "age": 36}
    assert markdown_request["messages"][0]["role"] == "system"
    assert "json_schema" in markdown_request["messages"][0]["content"]
    assert markdown_request["messages"][-1]["role"] == "user"
    assert markdown_items[-1].name == "Grace"
    assert markdown_items[-1].age == 40
    assert async_markdown_items[-1].model_dump() == {"name": "Grace", "age": 40}


def test_streaming_extension_and_parsed_result_variants() -> None:
    handler = MistralJSONSchemaHandler()

    class StreamingAnswer(BaseModel):
        answer: float

        @classmethod
        def from_streaming_response(
            cls, completion: Iterable[Any], stream_extractor: Any, **kwargs: Any
        ) -> Iterable[StreamingAnswer]:
            assert kwargs["context"] == {"request_id": "extension"}
            assert kwargs["strict"] is True
            return [
                cls.model_validate_json(chunk) for chunk in stream_extractor(completion)
            ]

    parsed_stream = handler._parse_streaming_response(
        StreamingAnswer,
        [event(content='{"answer": 2.5}')],
        {"request_id": "extension"},
        True,
    )
    iterable_model = IterableModel(User)
    iterable_result = iterable_model(tasks=[User(name="Ada", age=36)])
    parallel_model = ParallelBase(User, Answer)
    adapter_model = cast(type[BaseModel], ModelAdapter[int])
    adapter_result = adapter_model.model_validate({"content": 7})
    marker = response(content='{"answer": 2.5}')

    assert parsed_stream == [StreamingAnswer(answer=2.5)]
    assert handler._finalize_parsed_result(iterable_model, marker, iterable_result) == [
        User(name="Ada", age=36)
    ]
    assert (
        handler._finalize_parsed_result(parallel_model, marker, parsed_stream)
        is parsed_stream
    )
    assert handler._finalize_parsed_result(adapter_model, marker, adapter_result) == 7
    assert handler._finalize_parsed_result(Answer, marker, {"answer": 2.5}) == {
        "answer": 2.5
    }
    answer = Answer(answer=2.5)
    assert handler._finalize_parsed_result(Answer, marker, answer) is answer
    assert answer._raw_response is marker


@pytest.mark.parametrize(
    "handler",
    [MistralToolsHandler(), MistralJSONSchemaHandler(), MistralMDJSONHandler()],
)
def test_mistral_parsers_reject_responses_without_choices(
    handler: MistralToolsHandler | MistralJSONSchemaHandler | MistralMDJSONHandler,
) -> None:
    empty_response = response(content='{"name":"Ada","age":36}')
    empty_response.choices = []

    with pytest.raises(IndexError, match="list index out of range"):
        handler.parse_response(empty_response, User)


def test_mistral_message_conversion_and_pdf_rules() -> None:
    remote_pdf = PDF(source="https://example.test/report.pdf", data=None)
    messages = [{"role": "user", "content": ["Summarize", remote_pdf]}]

    tools_messages = MistralToolsHandler().convert_messages(messages)
    schema_messages = MistralJSONSchemaHandler().convert_messages(messages)
    markdown_messages = MistralMDJSONHandler().convert_messages(
        [{"role": "user", "content": "Summarize the report"}],
        autodetect_images=True,
    )

    expected_pdf = {
        "type": "document_url",
        "document_url": "https://example.test/report.pdf",
    }
    assert tools_messages[0]["content"][1] == expected_pdf
    assert schema_messages[0]["content"][1] == expected_pdf
    assert markdown_messages == [{"role": "user", "content": "Summarize the report"}]
    assert pdf_to_mistral(remote_pdf) == expected_pdf
    with pytest.raises(ValueError, match="only supports document URLs"):
        pdf_to_mistral(PDF(source="/tmp/report.pdf", data="cGRm"))
    with pytest.raises(ValueError, match="only supports document URLs"):
        pdf_to_mistral(PDF(source="https://example.test/report.pdf", data="cGRm"))
