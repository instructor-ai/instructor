"""Gemini v2 mode handlers."""

from __future__ import annotations

import inspect
import json
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterable as TypingIterable,
)
from typing import Any

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.partial import PartialBase
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.providers.gemini.utils import (
    handle_gemini_json,
    handle_gemini_tools,
    reask_gemini_json,
    reask_gemini_tools,
)
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler


class GeminiHandlerBase(ModeHandler):
    """Base handler for Gemini modes."""

    mode: Mode

    def extract_streaming_json(
        self, completion: TypingIterable[Any]
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from Gemini streaming responses."""
        for chunk in completion:
            try:
                if self.mode == Mode.TOOLS:
                    resp = chunk.candidates[0].content.parts[0].function_call
                    resp_dict = type(resp).to_dict(resp)  # type: ignore
                    if "args" in resp_dict:
                        yield json.dumps(resp_dict["args"])
                else:
                    try:
                        yield chunk.text
                    except Exception:
                        if chunk.candidates[0].content.parts[0].text:
                            yield chunk.candidates[0].content.parts[0].text
                            continue
                        raise
            except AttributeError:
                continue

    async def extract_streaming_json_async(
        self, completion: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Extract JSON chunks from Gemini async streams."""
        async for chunk in completion:
            try:
                if self.mode == Mode.TOOLS:
                    resp = chunk.candidates[0].content.parts[0].function_call
                    resp_dict = type(resp).to_dict(resp)  # type: ignore
                    if "args" in resp_dict:
                        yield json.dumps(resp_dict["args"])
                else:
                    try:
                        yield chunk.text
                    except Exception:
                        if chunk.candidates[0].content.parts[0].text:
                            yield chunk.candidates[0].content.parts[0].text
                            continue
                        raise
            except AttributeError:
                continue

    def _parse_streaming(
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

        if inspect.isasyncgen(response) or isinstance(response, AsyncIterator):
            return response_model.from_streaming_response_async(  # type: ignore[attr-defined]
                response,
                stream_extractor=self.extract_streaming_json_async,
                **parse_kwargs,
            )

        generator = response_model.from_streaming_response(  # type: ignore[attr-defined]
            response,
            stream_extractor=self.extract_streaming_json,
            **parse_kwargs,
        )
        if inspect.isclass(response_model) and issubclass(response_model, IterableBase):
            return generator
        if inspect.isclass(response_model) and issubclass(response_model, PartialBase):
            return list(generator)
        return list(generator)

    def _finalize(
        self,
        response_model: type[BaseModel],  # noqa: ARG002
        response: Any,
        parsed: Any,  # noqa: ARG002
    ) -> Any:
        if isinstance(parsed, AdapterBase):
            return parsed.content
        if isinstance(parsed, BaseModel):
            parsed._raw_response = response  # type: ignore[attr-defined]
        return parsed


@register_mode_handler(Provider.GEMINI, Mode.TOOLS)
class GeminiToolsHandler(GeminiHandlerBase):
    """Handler for Gemini TOOLS mode."""

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        return handle_gemini_tools(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_gemini_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if (
            stream
            and inspect.isclass(response_model)
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            return self._parse_streaming(
                response_model, response, validation_context, strict
            )
        parsed = response_model.parse_gemini_tools(  # type: ignore[attr-defined]
            response, validation_context, strict
        )
        return self._finalize(response_model, response, parsed)


@register_mode_handler(Provider.GEMINI, Mode.MD_JSON)
class GeminiJSONHandler(GeminiHandlerBase):
    """Handler for Gemini JSON mode."""

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        return handle_gemini_json(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_gemini_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if (
            stream
            and inspect.isclass(response_model)
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            return self._parse_streaming(
                response_model, response, validation_context, strict
            )
        parsed = response_model.parse_gemini_json(  # type: ignore[attr-defined]
            response, validation_context, strict
        )
        return self._finalize(response_model, response, parsed)


__all__ = ["GeminiToolsHandler", "GeminiJSONHandler"]
