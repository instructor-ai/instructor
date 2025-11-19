"""Anthropic v2 mode handlers with DSL-aware parsing."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Generator, Iterable as TypingIterable
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, get_origin
from weakref import WeakKeyDictionary

from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import Annotated

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anthropic.types import Message

from instructor import Mode, Provider
from instructor.core.exceptions import ConfigurationError, IncompleteOutputException
from instructor.dsl.iterable import IterableBase
from instructor.dsl.parallel import (
    AnthropicParallelBase,
    AnthropicParallelModel,
    ParallelBase,
    get_types_array,
    handle_anthropic_parallel_model,
)
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import AdapterBase
from instructor.processing.function_calls import extract_json_from_codeblock
from instructor.processing.multimodal import Audio, Image, PDF
from instructor.providers.anthropic.utils import (
    combine_system_messages,
    extract_system_messages,
    generate_anthropic_schema,
)
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler


def serialize_message_content(content: Any) -> Any:
    """Serialize message content, converting Pydantic models to dicts."""

    if isinstance(content, Image):
        source = str(content.source)
        if source.startswith(("http://", "https://")):
            return {
                "type": "image",
                "source": {"type": "url", "url": source},
            }
        if source.startswith("data:"):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content.media_type,
                    "data": content.data or source.split(",")[1],
                },
            }
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": content.media_type,
                "data": content.data or source,
            },
        }
    if isinstance(content, PDF):
        source = str(content.source)
        if source.startswith(("http://", "https://")):
            return {
                "type": "document",
                "source": {"type": "url", "url": source},
            }
        if source.startswith("data:"):
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": content.data or source.split(",")[1],
                },
            }
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": content.data or source,
            },
        }
    if isinstance(content, Audio):
        source = str(content.source)
        if source.startswith(("http://", "https://")):
            return {
                "type": "audio",
                "source": {"type": "url", "url": source},
            }
        return {
            "type": "audio",
            "source": {
                "type": "base64",
                "media_type": content.media_type,
                "data": content.data or source,
            },
        }
    if isinstance(content, str):
        return {"type": "text", "text": content}
    if isinstance(content, list):
        return [serialize_message_content(item) for item in content]
    if isinstance(content, dict):
        if "type" in content:
            return {k: serialize_message_content(v) for k, v in content.items()}
        return {k: serialize_message_content(v) for k, v in content.items()}
    if hasattr(content, "model_dump"):
        return content.model_dump()
    return content


def process_messages_for_anthropic(
    messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Process messages to serialize any Pydantic models in content."""

    processed: list[dict[str, Any]] = []
    for message in messages:
        msg_copy = message.copy()
        if "content" in msg_copy:
            content = msg_copy["content"]
            if isinstance(content, list):
                msg_copy["content"] = serialize_message_content(content)
            elif isinstance(content, (Image, Audio, PDF)) or hasattr(
                content, "model_dump"
            ):
                msg_copy["content"] = serialize_message_content(content)
        processed.append(msg_copy)
    return processed


class AnthropicHandlerBase(ModeHandler):
    """Common utilities for Anthropic handlers."""

    mode: Mode

    def __init__(self) -> None:
        self._streaming_models: WeakKeyDictionary[type[Any], None] = (
            WeakKeyDictionary()
        )

    def _register_streaming_from_kwargs(
        self, response_model: type[BaseModel] | None, kwargs: dict[str, Any]
    ) -> None:
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
        if response_model is None:
            return False
        if not inspect.isclass(response_model):
            return False
        if response_model in self._streaming_models:
            del self._streaming_models[response_model]
            return True
        return False

    def _parse_streaming_response(
        self,
        response_model: type[BaseModel],
        response: Any,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Any:
        parse_kwargs: dict[str, Any] = {}
        if validation_context is not None:
            parse_kwargs["context"] = validation_context
        if strict is not None:
            parse_kwargs["strict"] = strict

        if inspect.isasyncgen(response):
            return response_model.from_streaming_response_async(  # type: ignore[attr-defined]
                response,
                mode=self.mode,
                **parse_kwargs,
            )

        generator = response_model.from_streaming_response(  # type: ignore[attr-defined]
            response,
            mode=self.mode,
            **parse_kwargs,
        )
        return list(generator)

    def _finalize_parsed_result(
        self,
        response_model: type[BaseModel] | ParallelBase,
        response: Any,
        parsed: Any,
    ) -> Any:
        if isinstance(parsed, IterableBase):
            return [task for task in parsed.tasks]
        if isinstance(response_model, ParallelBase):
            return parsed
        if isinstance(parsed, AdapterBase):
            return parsed.content
        if isinstance(parsed, BaseModel):
            parsed._raw_response = response  # type: ignore[attr-defined]
        return parsed

    def _parse_with_callback(
        self,
        response: Any,
        response_model: type[BaseModel] | ParallelBase,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
        parser: Callable[
            [Any, type[BaseModel] | ParallelBase, dict[str, Any] | None, bool | None],
            Any,
        ],
    ) -> Any:
        if isinstance(response_model, type) and self._consume_streaming_flag(
            response_model
        ):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        parsed = parser(response, response_model, validation_context, strict)
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS)
class AnthropicToolsHandler(AnthropicHandlerBase):
    """Handler for Anthropic TOOLS mode."""

    mode = Mode.ANTHROPIC_TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        self._register_streaming_from_kwargs(response_model, kwargs)

        new_kwargs = kwargs.copy()
        system_messages = extract_system_messages(new_kwargs.get("messages", []))
        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m["role"] != "system"
        ]
        if "messages" in new_kwargs:
            new_kwargs["messages"] = process_messages_for_anthropic(
                new_kwargs["messages"]
            )

        if response_model is None:
            return None, new_kwargs

        is_parallel = False
        if get_origin(response_model) is TypingIterable:
            is_parallel = True

        if is_parallel:
            tool_schemas = handle_anthropic_parallel_model(response_model)
            new_kwargs["tools"] = tool_schemas
        else:
            tool_descriptions = generate_anthropic_schema(response_model)
            new_kwargs["tools"] = [tool_descriptions]

        if "tool_choice" not in new_kwargs:
            thinking_enabled = (
                "thinking" in new_kwargs
                and isinstance(new_kwargs.get("thinking"), dict)
                and new_kwargs.get("thinking", {}).get("type") == "enabled"
            )
            if thinking_enabled or is_parallel:
                new_kwargs["tool_choice"] = {"type": "auto"}
                if thinking_enabled:
                    new_kwargs["system"] = combine_system_messages(
                        new_kwargs.get("system"),
                        [
                            {
                                "type": "text",
                                "text": "Return only the tool call and no additional text.",
                            }
                        ],
                    )
            else:
                new_kwargs["tool_choice"] = {
                    "type": "tool",
                    "name": response_model.__name__,
                }

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Message,
        exception: Exception,
    ) -> dict[str, Any]:
        kwargs = kwargs.copy()

        assistant_content = []
        tool_use_id = None
        for content in response.content:
            assistant_content.append(content.model_dump())  # type: ignore[attr-defined]
            if content.type == "tool_use":
                tool_use_id = content.id

        reask_msgs = [{"role": "assistant", "content": assistant_content}]
        if tool_use_id is not None:
            reask_msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": (
                                "Validation Error found:\n"
                                f"{exception}\nRecall the function correctly, fix the errors"
                            ),
                            "is_error": True,
                        }
                    ],
                }
            )
        else:
            reask_msgs.append(
                {
                    "role": "user",
                    "content": (
                        "Validation Error due to no tool invocation:\n"
                        f"{exception}\nRecall the function correctly, fix the errors"
                    ),
                }
            )

        kwargs["messages"].extend(reask_msgs)
        return kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel] | ParallelBase,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> Any:
        return self._parse_with_callback(
            response,
            response_model,
            validation_context,
            strict,
            self._parse_tool_response,
        )

    def _parse_tool_response(
        self,
        response: Any,
        response_model: type[BaseModel] | ParallelBase,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Any:
        from anthropic.types import Message

        if isinstance(response, Message) and response.stop_reason == "max_tokens":
            raise IncompleteOutputException(last_completion=response)

        if isinstance(response_model, ParallelBase):
            return response_model.from_response(
                response,
                mode=self.mode,
                validation_context=validation_context,
                strict=strict,
            )

        origin = get_origin(response_model)
        if origin is TypingIterable:
            the_types = get_types_array(response_model)  # type: ignore[arg-type]
            type_registry = {t.__name__: t for t in the_types}

            def parallel_generator() -> Generator[BaseModel, None, None]:
                for content in response.content:
                    if getattr(content, "type", None) == "tool_use":
                        tool_name = content.name
                        if tool_name in type_registry:
                            model_class = type_registry[tool_name]
                            json_str = json.dumps(content.input)
                            yield model_class.model_validate_json(
                                json_str,
                                context=validation_context,
                                strict=strict,
                            )

            return parallel_generator()

        tool_calls = [
            json.dumps(c.input)
            for c in getattr(response, "content", [])
            if getattr(c, "type", None) == "tool_use"
        ]
        tool_calls_validator = TypeAdapter(
            Annotated[list[Any], Field(min_length=1, max_length=1)]
        )
        tool_call = tool_calls_validator.validate_python(tool_calls)[0]
        return response_model.model_validate_json(
            tool_call,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.ANTHROPIC, Mode.ANTHROPIC_REASONING_TOOLS)
class AnthropicReasoningToolsHandler(AnthropicToolsHandler):
    """Deprecated reasoning mode that delegates to AnthropicToolsHandler."""

    mode = Mode.ANTHROPIC_REASONING_TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        Mode.warn_anthropic_reasoning_tools_deprecation()
        return super().prepare_request(response_model, kwargs)


@register_mode_handler(Provider.ANTHROPIC, Mode.ANTHROPIC_PARALLEL_TOOLS)
class AnthropicParallelToolsHandler(AnthropicHandlerBase):
    """Handler for Anthropic parallel tool calling."""

    mode = Mode.ANTHROPIC_PARALLEL_TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[ParallelBase | None, dict[str, Any]]:
        self._register_streaming_from_kwargs(response_model, kwargs)

        new_kwargs = kwargs.copy()
        if new_kwargs.get("stream"):
            raise ConfigurationError(
                "stream=True is not supported when using ANTHROPIC_PARALLEL_TOOLS mode"
            )

        system_messages = extract_system_messages(new_kwargs.get("messages", []))
        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m["role"] != "system"
        ]

        if response_model is None:
            return None, new_kwargs

        new_kwargs["tools"] = handle_anthropic_parallel_model(response_model)
        new_kwargs["tool_choice"] = {"type": "auto"}

        if isinstance(response_model, AnthropicParallelBase):
            parallel_model: ParallelBase = response_model
        else:
            parallel_model = AnthropicParallelModel(typehint=response_model)

        return parallel_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Message,
        exception: Exception,
    ) -> dict[str, Any]:
        return AnthropicToolsHandler().handle_reask(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: ParallelBase,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> Any:
        return self._parse_with_callback(
            response,
            response_model,
            validation_context,
            strict,
            self._parse_parallel_response,
        )

    def _parse_parallel_response(
        self,
        response: Any,
        response_model: ParallelBase,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Any:
        return response_model.from_response(
            response,
            mode=self.mode,
            validation_context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.ANTHROPIC, Mode.ANTHROPIC_JSON)
class AnthropicJSONHandler(AnthropicHandlerBase):
    """Handler for Anthropic JSON mode."""

    mode = Mode.ANTHROPIC_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        self._register_streaming_from_kwargs(response_model, kwargs)

        new_kwargs = kwargs.copy()
        system_messages = extract_system_messages(new_kwargs.get("messages", []))
        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m["role"] != "system"
        ]
        if "messages" in new_kwargs:
            new_kwargs["messages"] = process_messages_for_anthropic(
                new_kwargs["messages"]
            )

        if response_model is None:
            return None, new_kwargs

        json_schema_message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n
            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )
        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"),
            [{"type": "text", "text": json_schema_message}],
        )
        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Message,
        exception: Exception,
    ) -> dict[str, Any]:
        kwargs = kwargs.copy()
        text_blocks = [c for c in response.content if c.type == "text"]
        if not text_blocks:
            text_content = "No text content found in response"
        else:
            text_content = text_blocks[-1].text
        reask_msg = {
            "role": "user",
            "content": (
                "Validation Errors found:\n"
                f"{exception}\nRecall the function correctly, fix the errors found in the following attempt:\n{text_content}"
            ),
        }
        kwargs["messages"].append(reask_msg)
        return kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> Any:
        return self._parse_with_callback(
            response,
            response_model,
            validation_context,
            strict,
            self._parse_json_response,
        )

    def _parse_json_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> BaseModel:
        from anthropic.types import Message
        from instructor.core.exceptions import ResponseParsingError

        if hasattr(response, "choices"):
            completion = response.choices[0]
            if completion.finish_reason == "length":
                raise IncompleteOutputException(last_completion=completion)
            text = completion.message.content
        else:
            if not isinstance(response, Message):
                raise ResponseParsingError(
                    "Response must be an Anthropic Message",
                    mode="JSON",
                    raw_response=response,
                )
            if response.stop_reason == "max_tokens":
                raise IncompleteOutputException(last_completion=response)
            text_blocks = [c for c in response.content if c.type == "text"]
            last_block = text_blocks[-1]
            text = re.sub(r"[\u0000-\u001F]", "", last_block.text)

        extra_text = extract_json_from_codeblock(text)
        if strict:
            return response_model.model_validate_json(
                extra_text,
                context=validation_context,
                strict=strict,
            )
        return response_model.model_validate_json(
            extra_text,
            context=validation_context,
        )


__all__ = [
    "AnthropicToolsHandler",
    "AnthropicReasoningToolsHandler",
    "AnthropicParallelToolsHandler",
    "AnthropicJSONHandler",
]
