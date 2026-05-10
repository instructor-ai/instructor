"""Mistral v2 mode handlers.

This module implements mode handlers for Mistral AI using the v2 registry system.
Supports TOOLS, JSON_SCHEMA, and MD_JSON modes.

Mistral has its own API format that differs from OpenAI:
- Uses `chat.complete()` and `chat.complete_async()` instead of `chat.completions.create()`
- Uses `chat.stream()` and `chat.stream_async()` for streaming
- Tool calling uses `tool_choice="any"` instead of specific tool selection
- Structured outputs use `response_format_from_pydantic_model()` helper
"""

from __future__ import annotations

import inspect
import json
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterable as TypingIterable,
)
from textwrap import dedent
from typing import TYPE_CHECKING, Any, get_origin
from weakref import WeakKeyDictionary

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import ParallelBase, get_types_array
from instructor.v2.dsl.partial import PartialBase
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.core.multimodal import convert_messages as convert_messages_v1
from instructor.v2.core.json import (
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
)
from instructor.v2.providers.openai.schema import generate_openai_schema
from instructor.v2.core.messages import dump_message, merge_consecutive_messages
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler


class MistralHandlerBase(ModeHandler):
    """Base class for Mistral handlers with shared utilities."""

    mode: Mode

    def __init__(self) -> None:
        self._streaming_models: WeakKeyDictionary[type[Any], None] = WeakKeyDictionary()

    def _register_streaming_from_kwargs(
        self, response_model: type[BaseModel] | None, kwargs: dict[str, Any]
    ) -> None:
        """Register model for streaming if stream=True in kwargs."""
        if response_model is None:
            return
        if kwargs.get("stream"):
            self.mark_streaming_model(response_model, True)

    def mark_streaming_model(
        self, response_model: type[BaseModel] | None, stream: bool
    ) -> None:
        """Record that the response model expects streaming output."""
        if not stream or response_model is None:
            return
        if inspect.isclass(response_model) and issubclass(
            response_model, (IterableBase, PartialBase)
        ):
            self._streaming_models[response_model] = None

    def _consume_streaming_flag(
        self, response_model: type[BaseModel] | ParallelBase | None
    ) -> bool:
        """Check and consume streaming flag for a model."""
        if response_model is None:
            return False
        if not inspect.isclass(response_model):
            return False
        if response_model in self._streaming_models:
            del self._streaming_models[response_model]
            return True
        return False

    def extract_streaming_json(
        self, completion: TypingIterable[Any]
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from Mistral streaming responses."""

        def _raw_chunks() -> Generator[str, None, None]:
            for chunk in completion:
                try:
                    if self.mode == Mode.TOOLS:
                        if not chunk.data.choices[0].delta.tool_calls:
                            continue
                        yield (
                            chunk.data.choices[0].delta.tool_calls[0].function.arguments
                        )
                    else:
                        yield chunk.data.choices[0].delta.content
                except AttributeError:
                    continue

        raw_chunks = _raw_chunks()
        if self.mode == Mode.MD_JSON:
            yield from extract_json_from_stream(raw_chunks)
            return
        yield from raw_chunks

    async def extract_streaming_json_async(
        self, completion: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Extract JSON chunks from Mistral async streams."""

        async def _raw_chunks() -> AsyncGenerator[str, None]:
            async for chunk in completion:
                try:
                    if self.mode == Mode.TOOLS:
                        if not chunk.data.choices[0].delta.tool_calls:
                            continue
                        yield (
                            chunk.data.choices[0].delta.tool_calls[0].function.arguments
                        )
                    else:
                        yield chunk.data.choices[0].delta.content
                except AttributeError:
                    continue

        raw_chunks = _raw_chunks()
        if self.mode == Mode.MD_JSON:
            async for chunk in extract_json_from_stream_async(raw_chunks):
                yield chunk
            return
        async for chunk in raw_chunks:
            yield chunk

    def convert_messages(
        self, messages: list[dict[str, Any]], autodetect_images: bool = False
    ) -> list[dict[str, Any]]:
        """Convert messages for Mistral-compatible multimodal payloads."""
        if self.mode == Mode.TOOLS:
            target_mode = Mode.MISTRAL_TOOLS
        elif self.mode == Mode.JSON_SCHEMA:
            target_mode = Mode.MISTRAL_STRUCTURED_OUTPUTS
        else:
            target_mode = Mode.MD_JSON
        return convert_messages_v1(
            messages, target_mode, autodetect_images=autodetect_images
        )

    def _parse_streaming_response(
        self,
        response_model: type[BaseModel],
        response: Any,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Any:
        """Parse a streaming response using DSL methods."""
        parse_kwargs: dict[str, Any] = {}
        if validation_context is not None:
            parse_kwargs["context"] = validation_context
        if strict is not None:
            parse_kwargs["strict"] = strict

        task_parser = None
        if (
            self.mode == Mode.TOOLS
            and inspect.isclass(response_model)
            and issubclass(response_model, IterableBase)
        ):
            task_parser = response_model.tasks_from_task_list_chunks  # type: ignore[attr-defined]

        if inspect.isasyncgen(response) or isinstance(response, AsyncIterator):
            return response_model.from_streaming_response_async(  # type: ignore[attr-defined]
                response,
                stream_extractor=self.extract_streaming_json_async,
                task_parser=(
                    response_model.tasks_from_task_list_chunks_async  # type: ignore[attr-defined]
                    if task_parser is not None
                    else None
                ),
                **parse_kwargs,
            )

        generator = response_model.from_streaming_response(  # type: ignore[attr-defined]
            response,
            stream_extractor=self.extract_streaming_json,
            task_parser=task_parser,
            **parse_kwargs,
        )
        if inspect.isclass(response_model) and issubclass(response_model, IterableBase):
            return generator
        if inspect.isclass(response_model) and issubclass(response_model, PartialBase):
            return list(generator)
        return list(generator)

    def _finalize_parsed_result(
        self,
        response_model: type[BaseModel] | ParallelBase,
        response: Any,
        parsed: Any,
    ) -> Any:
        """Finalize parsed result, handling DSL types."""
        if isinstance(parsed, IterableBase):
            return [task for task in parsed.tasks]
        if isinstance(response_model, ParallelBase):
            return parsed
        if isinstance(parsed, AdapterBase):
            return parsed.content
        if isinstance(parsed, BaseModel):
            parsed._raw_response = response  # type: ignore[attr-defined]
        return parsed

    def _extract_tool_call_json(self, response: Any) -> str:
        """Extract JSON from tool call response.

        Mistral returns tool call arguments as either a string or a dict,
        so we need to handle both cases.
        """
        tool_call = response.choices[0].message.tool_calls[0]
        args = tool_call.function.arguments
        if isinstance(args, dict):
            return json.dumps(args)
        return args

    def _extract_text_content(self, response: Any) -> str:
        """Extract text content from response."""
        return response.choices[0].message.content or ""


@register_mode_handler(Provider.MISTRAL, Mode.TOOLS)
class MistralToolsHandler(MistralHandlerBase):
    """Handler for Mistral TOOLS mode.

    Uses Mistral's tool calling API with tool_choice="any".
    """

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with tool definitions."""
        new_kwargs = kwargs.copy()

        if response_model is None:
            return None, new_kwargs

        # Detect if this is a parallel tools request (Iterable[Union[...]])
        # When streaming, treat Iterable[T] as streaming instead of parallel tools.
        origin = get_origin(response_model)
        is_parallel = origin is TypingIterable and not new_kwargs.get("stream")

        # Prepare response model: wrap simple types in ModelAdapter
        if not is_parallel:
            from instructor.v2.core.response_model import prepare_response_model

            response_model = prepare_response_model(response_model)

        self._register_streaming_from_kwargs(response_model, new_kwargs)

        if is_parallel:
            # Handle parallel model - generate tools for each type
            the_types = get_types_array(response_model)  # type: ignore[arg-type]
            tools = []
            for model_type in the_types:
                schema = generate_openai_schema(model_type)
                tools.append({"type": "function", "function": schema})
            new_kwargs["tools"] = tools
        else:
            schema = generate_openai_schema(response_model)
            new_kwargs["tools"] = [{"type": "function", "function": schema}]

        # Mistral uses tool_choice="any" to force tool use
        new_kwargs["tool_choice"] = "any"

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for tools mode."""
        kwargs = kwargs.copy()
        reask_msgs = [dump_message(response.choices[0].message)]

        for tool_call in response.choices[0].message.tool_calls:
            reask_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": (
                        f"Validation Error found:\n{exception}\n"
                        "Recall the function correctly, fix the errors"
                    ),
                }
            )

        kwargs["messages"].extend(reask_msgs)
        return kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse tool call response."""
        # Check for streaming
        consume_streaming = isinstance(
            response_model, type
        ) and self._consume_streaming_flag(response_model)
        if consume_streaming:
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Check for incomplete output
        if hasattr(response, "choices") and response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        # Handle parallel tools (Iterable[Union[...]])
        origin = get_origin(response_model)
        if origin is TypingIterable:
            the_types = get_types_array(response_model)  # type: ignore[arg-type]
            type_registry = {t.__name__: t for t in the_types}

            def parallel_generator() -> Generator[BaseModel, None, None]:
                for tool_call in response.choices[0].message.tool_calls:
                    name = tool_call.function.name
                    if name in type_registry:
                        model_class = type_registry[name]
                        args = tool_call.function.arguments
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        yield model_class.model_validate_json(
                            args,
                            context=validation_context,
                            strict=strict,
                        )

            return parallel_generator()

        # Standard tool call parsing
        json_str = self._extract_tool_call_json(response)
        parsed = response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(Provider.MISTRAL, Mode.JSON_SCHEMA)
class MistralJSONSchemaHandler(MistralHandlerBase):
    """Handler for Mistral structured outputs (JSON_SCHEMA mode).

    Uses Mistral's native structured outputs via response_format parameter.
    Requires the mistralai SDK's response_format_from_pydantic_model helper.
    """

    mode = Mode.JSON_SCHEMA

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with JSON schema response format."""
        self._register_streaming_from_kwargs(response_model, kwargs)

        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()

        # Use Mistral's helper to create response format
        from mistralai.extra import response_format_from_pydantic_model

        new_kwargs["response_format"] = response_format_from_pydantic_model(
            response_model
        )

        # Remove any tool-related kwargs
        new_kwargs.pop("tools", None)
        new_kwargs.pop("tool_choice", None)

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for JSON schema mode."""
        kwargs = kwargs.copy()
        reask_msgs = [
            {
                "role": "assistant",
                "content": response.choices[0].message.content,
            },
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors"
                ),
            },
        ]
        kwargs["messages"].extend(reask_msgs)
        return kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse JSON schema response."""
        # Check for streaming
        if isinstance(response_model, type) and self._consume_streaming_flag(
            response_model
        ):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Check for incomplete output
        if hasattr(response, "choices") and response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        text = self._extract_text_content(response)
        parsed = response_model.model_validate_json(
            text,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(Provider.MISTRAL, Mode.MD_JSON)
class MistralMDJSONHandler(MistralHandlerBase):
    """Handler for MD_JSON mode - extract JSON from markdown code blocks."""

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with JSON schema instruction in messages."""
        self._register_streaming_from_kwargs(response_model, kwargs)

        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()
        schema = response_model.model_json_schema()

        message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n

            {json.dumps(schema, indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

        # Add system message with schema
        messages = new_kwargs.get("messages", [])
        if messages and messages[0]["role"] != "system":
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": message,
                },
            )
        elif messages and isinstance(messages[0]["content"], str):
            messages[0]["content"] += f"\n\n{message}"
        elif (
            messages
            and isinstance(messages[0]["content"], list)
            and messages[0]["content"]
        ):
            messages[0]["content"][0]["text"] += f"\n\n{message}"
        else:
            messages.insert(0, {"role": "system", "content": message})

        # Add user message requesting JSON in code block
        messages.append(
            {
                "role": "user",
                "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
            },
        )
        new_kwargs["messages"] = merge_consecutive_messages(messages)

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for MD_JSON mode."""
        kwargs = kwargs.copy()
        reask_msgs = [
            {
                "role": "assistant",
                "content": response.choices[0].message.content,
            },
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors"
                ),
            },
        ]
        kwargs["messages"].extend(reask_msgs)
        return kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse JSON from markdown code block in response."""
        # Check for streaming
        if isinstance(response_model, type) and self._consume_streaming_flag(
            response_model
        ):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Check for incomplete output
        if hasattr(response, "choices") and response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        text = self._extract_text_content(response)
        json_str = extract_json_from_codeblock(text)
        parsed = response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


__all__ = [
    "MistralToolsHandler",
    "MistralJSONSchemaHandler",
    "MistralMDJSONHandler",
]
