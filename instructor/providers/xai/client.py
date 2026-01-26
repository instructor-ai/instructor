from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast, overload
import json

from instructor.dsl.iterable import IterableBase
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import AdapterBase

from instructor.utils.core import prepare_response_model
from pydantic import BaseModel

import instructor
from .utils import _convert_messages


def _get_model_schema(response_model: Any) -> dict[str, Any]:
    """
    Safely get JSON schema from a response model.

    Handles both regular models and wrapped types by checking for the
    model_json_schema method with hasattr.

    Args:
        response_model: The response model (may be regular or wrapped)

    Returns:
        The JSON schema dictionary
    """
    if hasattr(response_model, "model_json_schema") and callable(
        response_model.model_json_schema
    ):
        schema_method = response_model.model_json_schema
        return schema_method()
    return {}


def _get_model_name(response_model: Any) -> str:
    """
    Safely get the name of a response model.

    Args:
        response_model: The response model

    Returns:
        The model name or 'Model' as fallback
    """
    return getattr(response_model, "__name__", "Model")


def _finalize_parsed_response(parsed: Any, raw_response: Any) -> Any:
    if isinstance(parsed, BaseModel):
        parsed._raw_response = raw_response
    if isinstance(parsed, IterableBase):
        return [task for task in parsed.tasks]
    if isinstance(parsed, AdapterBase):
        return parsed.content
    return parsed


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
    valid_modes = {instructor.Mode.XAI_JSON, instructor.Mode.XAI_TOOLS}

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
            resp = await chat.sample()  # type: ignore[misc]
            return resp

        assert response_model is not None

        prepared_model = response_model
        if mode == instructor.Mode.XAI_TOOLS or is_stream:
            prepared_model = prepare_response_model(response_model)
        assert prepared_model is not None

        if mode == instructor.Mode.XAI_JSON:
            if is_stream:
                # code from xai_sdk.chat.parse
                chat.proto.response_format.CopyFrom(
                    xchat.chat_pb2.ResponseFormat(
                        format_type=xchat.chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
                        schema=json.dumps(_get_model_schema(prepared_model)),
                    )
                )
                json_chunks = (chunk.content async for _, chunk in chat.stream())  # type: ignore[misc]
                # response_model is guaranteed to be a type[BaseModel] at this point due to earlier assertion
                rm = cast(type[BaseModel], prepared_model)
                if issubclass(rm, IterableBase):
                    return rm.tasks_from_chunks_async(json_chunks)  # type: ignore
                elif issubclass(rm, PartialBase):
                    return rm.model_from_chunks_async(json_chunks)  # type: ignore
                else:
                    raise ValueError(
                        f"Unsupported response model type for streaming: {_get_model_name(response_model)}"
                    )
            else:
                raw, parsed = await chat.parse(response_model)  # type: ignore[misc]
                parsed._raw_response = raw
                return parsed
        else:
            tool_obj = xchat.tool(
                name=_get_model_name(prepared_model),
                description=prepared_model.__doc__ or "",
                parameters=_get_model_schema(prepared_model),
            )
            chat.proto.tools.append(tool_obj)  # type: ignore[arg-type]
            tool_name = tool_obj.function.name  # type: ignore[attr-defined]
            chat.proto.tool_choice.CopyFrom(xchat.required_tool(tool_name))
            if is_stream:
                stream_iter = chat.stream()  # type: ignore[misc]
                args = (
                    resp.tool_calls[0].function.arguments  # type: ignore[index,attr-defined]
                    async for resp, _ in stream_iter  # type: ignore[assignment]
                    if resp.tool_calls and resp.finish_reason == "REASON_INVALID"  # type: ignore[attr-defined]
                )
                rm = cast(type[BaseModel], prepared_model)
                if issubclass(rm, IterableBase):
                    return rm.tasks_from_chunks_async(args)  # type: ignore
                elif issubclass(rm, PartialBase):
                    return rm.model_from_chunks_async(args)  # type: ignore
                else:
                    raise ValueError(
                        f"Unsupported response model type for streaming: {_get_model_name(response_model)}"
                    )
            else:
                resp = await chat.sample()  # type: ignore[misc]
                if not resp.tool_calls:  # type: ignore[attr-defined]
                    # If no tool calls, try to extract from text content
                    from ...processing.function_calls import _validate_model_from_json
                    from ...utils import extract_json_from_codeblock

                    # Try to extract JSON from text content
                    text_content: str = ""
                    if hasattr(resp, "text") and resp.text:  # type: ignore[attr-defined]
                        text_content = str(resp.text)  # type: ignore[attr-defined]
                    elif hasattr(resp, "content") and resp.content:  # type: ignore[attr-defined]
                        content = resp.content  # type: ignore[attr-defined]
                        if isinstance(content, str):
                            text_content = content
                        elif isinstance(content, list) and content:
                            text_content = str(content[0])

                    if text_content:
                        json_str = extract_json_from_codeblock(text_content)
                        model_for_validation = cast(type[Any], prepared_model)
                        parsed = _validate_model_from_json(
                            model_for_validation, json_str, None, strict
                        )
                        return _finalize_parsed_response(parsed, resp)

                    raise ValueError(
                        f"No tool calls returned from xAI and no text content available. "
                        f"Response: {resp}"
                    )

                args = resp.tool_calls[0].function.arguments  # type: ignore[index,attr-defined]
                from ...processing.function_calls import _validate_model_from_json

                model_for_validation = cast(type[Any], prepared_model)
                parsed = _validate_model_from_json(
                    model_for_validation, args, None, strict
                )
                return _finalize_parsed_response(parsed, resp)

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
            resp = chat.sample()  # type: ignore[misc]
            return resp

        assert response_model is not None

        prepared_model = response_model
        if mode == instructor.Mode.XAI_TOOLS or is_stream:
            prepared_model = prepare_response_model(response_model)
        assert prepared_model is not None

        if mode == instructor.Mode.XAI_JSON:
            if is_stream:
                # code from xai_sdk.chat.parse
                chat.proto.response_format.CopyFrom(
                    xchat.chat_pb2.ResponseFormat(
                        format_type=xchat.chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
                        schema=json.dumps(_get_model_schema(prepared_model)),
                    )
                )
                json_chunks = (chunk.content for _, chunk in chat.stream())  # type: ignore[misc]
                rm = cast(type[BaseModel], prepared_model)
                if issubclass(rm, IterableBase):
                    return rm.tasks_from_chunks(json_chunks)
                elif issubclass(rm, PartialBase):
                    return rm.model_from_chunks(json_chunks)
                else:
                    raise ValueError(
                        f"Unsupported response model type for streaming: {_get_model_name(response_model)}"
                    )
            else:
                raw, parsed = chat.parse(response_model)  # type: ignore[misc]
                parsed._raw_response = raw
                return parsed
        else:
            tool_obj = xchat.tool(
                name=_get_model_name(prepared_model),
                description=prepared_model.__doc__ or "",
                parameters=_get_model_schema(prepared_model),
            )
            chat.proto.tools.append(tool_obj)  # type: ignore[arg-type]
            tool_name = tool_obj.function.name  # type: ignore[attr-defined]
            chat.proto.tool_choice.CopyFrom(xchat.required_tool(tool_name))
            if is_stream:
                stream_iter = chat.stream()  # type: ignore[misc]
                for resp, _ in stream_iter:  # type: ignore[assignment]
                    # For xAI, tool_calls are returned at the end of the response.
                    # Effectively, it is not a streaming response.
                    # See: https://docs.x.ai/docs/guides/function-calling
                    if resp.tool_calls:  # type: ignore[attr-defined]
                        args = resp.tool_calls[0].function.arguments  # type: ignore[index,attr-defined]
                        rm = cast(type[BaseModel], prepared_model)
                        if issubclass(rm, IterableBase):
                            return rm.tasks_from_chunks(args)
                        elif issubclass(rm, PartialBase):
                            return rm.model_from_chunks(args)
                        else:
                            raise ValueError(
                                f"Unsupported response model type for streaming: {_get_model_name(response_model)}"
                            )
            else:
                resp = chat.sample()  # type: ignore[misc]
                if not resp.tool_calls:  # type: ignore[attr-defined]
                    # If no tool calls, try to extract from text content
                    from ...processing.function_calls import _validate_model_from_json
                    from ...utils import extract_json_from_codeblock

                    # Try to extract JSON from text content
                    text_content: str = ""
                    if hasattr(resp, "text") and resp.text:  # type: ignore[attr-defined]
                        text_content = str(resp.text)  # type: ignore[attr-defined]
                    elif hasattr(resp, "content") and resp.content:  # type: ignore[attr-defined]
                        content = resp.content  # type: ignore[attr-defined]
                        if isinstance(content, str):
                            text_content = content
                        elif isinstance(content, list) and content:
                            text_content = str(content[0])

                    if text_content:
                        json_str = extract_json_from_codeblock(text_content)
                        model_for_validation = cast(type[Any], prepared_model)
                        parsed = _validate_model_from_json(
                            model_for_validation, json_str, None, strict
                        )
                        return _finalize_parsed_response(parsed, resp)

                    raise ValueError(
                        f"No tool calls returned from xAI and no text content available. "
                        f"Response: {resp}"
                    )

                args = resp.tool_calls[0].function.arguments  # type: ignore[index,attr-defined]
                from ...processing.function_calls import _validate_model_from_json

                model_for_validation = cast(type[Any], prepared_model)
                parsed = _validate_model_from_json(
                    model_for_validation, args, None, strict
                )
                return _finalize_parsed_response(parsed, resp)

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
