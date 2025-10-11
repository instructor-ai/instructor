from __future__ import annotations

from typing import Any, overload
import json

from instructor.dsl.iterable import IterableBase
from instructor.dsl.partial import PartialBase

from instructor.utils.core import prepare_response_model
from pydantic import BaseModel

import instructor
from .utils import _convert_messages

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xai_sdk.sync.client import Client as SyncClient
    from xai_sdk.aio.client import Client as AsyncClient
    from xai_sdk import chat as xchat
else:
    try:
        from xai_sdk.sync.client import Client as SyncClient
        from xai_sdk.aio.client import Client as AsyncClient
        from xai_sdk import chat as xchat
    except ImportError:
        SyncClient = None
        AsyncClient = None
        xchat = None


@overload
def from_xai(
    client: SyncClient,
    mode: instructor.Mode = instructor.Mode.XAI_JSON,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_xai(
    client: AsyncClient,
    mode: instructor.Mode = instructor.Mode.XAI_JSON,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_xai(
    client: SyncClient | AsyncClient,
    mode: instructor.Mode = instructor.Mode.XAI_JSON,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    valid_modes = {
        instructor.Mode.XAI_JSON,
        instructor.Mode.XAI_TOOLS,
        instructor.Mode.MD_YAML,
    }

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="xAI", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, (SyncClient, AsyncClient)):
        from ...core.exceptions import ClientError

        raise ClientError(
            "Client must be an instance of xai_sdk.sync.client.Client or xai_sdk.aio.client.Client. "
            f"Got: {type(client).__name__}"
        )

    async def acreate(
        response_model: type[BaseModel] | None,
        messages: list[dict[str, Any]],
        strict: bool = True,
        **call_kwargs: Any,
    ):
        x_messages = _convert_messages(messages)
        model = call_kwargs.pop("model")
        # Remove instructor-specific kwargs that xAI doesn't support
        call_kwargs.pop("max_retries", None)
        call_kwargs.pop("validation_context", None)
        call_kwargs.pop("context", None)
        call_kwargs.pop("hooks", None)
        is_stream = call_kwargs.pop("stream", False)

        chat = client.chat.create(model=model, messages=x_messages, **call_kwargs)

        if response_model is None:
            resp = await chat.sample()
            return resp

        if is_stream:
            response_model = prepare_response_model(response_model)

        if mode == instructor.Mode.XAI_JSON:
            if is_stream:
                # code from xai_sdk.chat.parse
                chat.proto.response_format.CopyFrom(
                    xchat.chat_pb2.ResponseFormat(
                        format_type=xchat.chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
                        schema=json.dumps(response_model.model_json_schema()),
                    )
                )
                json_chunks = (chunk.content async for _, chunk in chat.stream())
                if issubclass(response_model, IterableBase):
                    return response_model.tasks_from_chunks_async(json_chunks)
                elif issubclass(response_model, PartialBase):
                    return response_model.model_from_chunks_async(json_chunks)
                else:
                    raise ValueError(
                        f"Unsupported response model type for streaming: {response_model.__name__}"
                    )
            else:
                raw, parsed = await chat.parse(response_model)
                parsed._raw_response = raw
                return parsed
        else:
            tool = xchat.tool(
                name=response_model.__name__,
                description=response_model.__doc__ or "",
                parameters=response_model.model_json_schema(),
            )
            chat.proto.tools.append(tool)
            chat.proto.tool_choice.mode = xchat.chat_pb2.ToolMode.TOOL_MODE_AUTO
            if is_stream:
                args = (
                    resp.tool_calls[0].function.arguments
                    async for resp, _ in chat.stream()
                    if resp.tool_calls and resp.finish_reason == "REASON_INVALID"
                )
                if issubclass(response_model, IterableBase):
                    return response_model.tasks_from_chunks_async(args)
                elif issubclass(response_model, PartialBase):
                    return response_model.model_from_chunks_async(args)
                else:
                    raise ValueError(
                        f"Unsupported response model type for streaming: {response_model.__name__}"
                    )
            else:
                resp = await chat.sample()
                args = resp.tool_calls[0].function.arguments
                from ...processing.function_calls import _validate_model_from_json

                parsed = _validate_model_from_json(response_model, args, None, strict)
                parsed._raw_response = resp
                return parsed

    def create(
        response_model: type[BaseModel] | None,
        messages: list[dict[str, Any]],
        strict: bool = True,
        **call_kwargs: Any,
    ):
        x_messages = _convert_messages(messages)
        model = call_kwargs.pop("model")
        # Remove instructor-specific kwargs that xAI doesn't support
        call_kwargs.pop("max_retries", None)
        call_kwargs.pop("validation_context", None)
        call_kwargs.pop("context", None)
        call_kwargs.pop("hooks", None)
        # Check if streaming is requested
        is_stream = call_kwargs.pop("stream", False)

        chat = client.chat.create(model=model, messages=x_messages, **call_kwargs)

        if response_model is None:
            resp = chat.sample()
            return resp

        if is_stream:
            response_model = prepare_response_model(response_model)

        if mode == instructor.Mode.XAI_JSON:
            if is_stream:
                # code from xai_sdk.chat.parse
                chat.proto.response_format.CopyFrom(
                    xchat.chat_pb2.ResponseFormat(
                        format_type=xchat.chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
                        schema=json.dumps(response_model.model_json_schema()),
                    )
                )
                json_chunks = (chunk.content for _, chunk in chat.stream())
                if issubclass(response_model, IterableBase):
                    return response_model.tasks_from_chunks(json_chunks)
                elif issubclass(response_model, PartialBase):
                    return response_model.model_from_chunks(json_chunks)
                else:
                    raise ValueError(
                        f"Unsupported response model type for streaming: {response_model.__name__}"
                    )
            else:
                raw, parsed = chat.parse(response_model)
                parsed._raw_response = raw
                return parsed
        else:
            tool = xchat.tool(
                name=response_model.__name__,
                description=response_model.__doc__ or "",
                parameters=response_model.model_json_schema(),
            )
            chat.proto.tools.append(tool)
            chat.proto.tool_choice.mode = xchat.chat_pb2.ToolMode.TOOL_MODE_AUTO
            if is_stream:
                for resp, _ in chat.stream():
                    # For xAI, tool_calls are returned at the end of the response.
                    # Effectively, it is not a streaming response.
                    # See: https://docs.x.ai/docs/guides/function-calling
                    if resp.tool_calls:
                        args = resp.tool_calls[0].function.arguments
                        if issubclass(response_model, IterableBase):
                            return response_model.tasks_from_chunks(args)
                        elif issubclass(response_model, PartialBase):
                            return response_model.model_from_chunks(args)
                        else:
                            raise ValueError(
                                f"Unsupported response model type for streaming: {response_model.__name__}"
                            )
            else:
                resp = chat.sample()
                args = resp.tool_calls[0].function.arguments
                from ...processing.function_calls import _validate_model_from_json

                parsed = _validate_model_from_json(response_model, args, None, strict)
                parsed._raw_response = resp
                return parsed

    if isinstance(client, AsyncClient):
        return instructor.AsyncInstructor(
            client=client,
            create=acreate,
            provider=instructor.Provider.XAI,
            mode=mode,
            **kwargs,
        )
    else:
        return instructor.Instructor(
            client=client,
            create=create,
            provider=instructor.Provider.XAI,
            mode=mode,
            **kwargs,
        )
