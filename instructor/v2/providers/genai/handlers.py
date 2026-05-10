from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast

from pydantic import BaseModel

from ....dsl.iterable import IterableBase
from ....dsl.parallel import ParallelBase
from ....dsl.partial import Partial, PartialBase
from ....dsl.simple_type import AdapterBase
from ....processing.multimodal import extract_genai_multimodal_content
from ...providers.gemini import utils as gemini_utils
from ...core.response_model import prepare_response_model
from ...core.decorators import register_mode_handler
from ...core.handler import ModeHandler
from ....mode import Mode
from ....utils.providers import Provider


class GenAIHandlerBase(ModeHandler):
    """Common utilities shared across GenAI mode handlers."""

    def __init__(self, mode: Mode | None = None) -> None:
        """Initialize handler with optional mode."""
        self.mode = mode

    def _clone_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs.copy()

    def _pop_autodetect_images(self, kwargs: dict[str, Any]) -> bool:
        return bool(kwargs.pop("autodetect_images", False))

    def _extract_system_instruction(self, kwargs: dict[str, Any]) -> str | None:
        if "system" in kwargs and kwargs["system"] is not None:
            return cast(str, kwargs.pop("system"))
        if "messages" in kwargs:
            return gemini_utils.extract_genai_system_message(
                cast(list[dict[str, Any]], kwargs["messages"])
            )
        return None

    def extract_streaming_json(self, completion: Any) -> Generator[str, None, None]:
        """Extract JSON chunks from GenAI streaming responses."""
        for chunk in completion:
            try:
                if self.mode == Mode.TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
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
        """Extract JSON chunks from GenAI async streams."""
        async for chunk in completion:
            try:
                if self.mode == Mode.TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
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

    def _wrap_streaming_model(
        self,
        response_model: type[BaseModel] | None,
        stream: bool,
    ) -> type[BaseModel] | None:
        if response_model is None:
            return None
        if (
            stream
            and isinstance(response_model, type)
            and not issubclass(response_model, PartialBase)
        ):
            return Partial[response_model]  # type: ignore[return-value]
        return response_model

    def _convert_messages_to_contents(
        self,
        kwargs: dict[str, Any],
        autodetect_images: bool,
    ) -> dict[str, Any]:
        contents = gemini_utils.convert_to_genai_messages(kwargs.get("messages", []))
        kwargs["contents"] = extract_genai_multimodal_content(
            contents, autodetect_images
        )
        kwargs.pop("messages", None)
        return kwargs

    def _cleanup_provider_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        # Keep 'model' as it's required by GenAI API
        # Remove other OpenAI-specific params that should be in config
        for key in (
            "response_model",
            "generation_config",
            "safety_settings",
            "thinking_config",
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "kwargs",  # Remove any nested kwargs key
        ):
            kwargs.pop(key, None)
        return kwargs

    def _prepare_without_response_model(
        self,
        kwargs: dict[str, Any],
        autodetect_images: bool,
    ) -> dict[str, Any]:
        from google.genai import types

        system_instruction = self._extract_system_instruction(kwargs)
        kwargs = self._convert_messages_to_contents(kwargs, autodetect_images)
        if system_instruction:
            kwargs["config"] = types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        return self._cleanup_provider_kwargs(kwargs)

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        raise NotImplementedError

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel] | None,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,
    ) -> BaseModel | Any:
        if response_model is None:
            return response

        if (
            stream
            and isinstance(response_model, type)
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            if is_async:
                return response_model.from_streaming_response_async(  # type: ignore
                    response,
                    stream_extractor=self.extract_streaming_json_async,
                )
            generator = response_model.from_streaming_response(  # type: ignore
                response,
                stream_extractor=self.extract_streaming_json,
            )
            if issubclass(response_model, IterableBase):
                return generator
            return list(generator)

        if self.mode == Mode.TOOLS:
            model = response_model.parse_genai_tools(  # type: ignore[attr-defined]
                response,
                validation_context,
                strict,
            )
        else:
            model = response_model.parse_genai_structured_outputs(  # type: ignore[attr-defined]
                response,
                validation_context,
                strict,
            )

        if isinstance(model, IterableBase):
            return list(model.tasks)

        if isinstance(response_model, ParallelBase):
            return model

        if isinstance(model, AdapterBase):
            return model.content

        model._raw_response = response  # type: ignore[attr-defined]
        return model

    def handle_reask(
        self,
        *,
        kwargs: dict[str, Any],
        response: Any,  # noqa: ARG002
        exception: Exception,  # noqa: ARG002
        failed_attempts: list[Any] | None = None,  # noqa: ARG002  # noqa: ARG002
    ) -> dict[str, Any]:
        return kwargs.copy()


@register_mode_handler(Provider.GENAI, Mode.TOOLS)
class GenAIToolsHandler(GenAIHandlerBase):
    """Mode handler for GenAI tools/function calling."""

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        from google.genai import types

        new_kwargs = self._clone_kwargs(kwargs)
        autodetect_images = self._pop_autodetect_images(new_kwargs)
        stream = bool(new_kwargs.get("stream", False))

        prepared_model = prepare_response_model(response_model)
        if prepared_model is None:
            return None, self._prepare_without_response_model(
                new_kwargs, autodetect_images
            )

        prepared_model = self._wrap_streaming_model(prepared_model, stream)
        schema = gemini_utils.map_to_genai_schema(
            gemini_utils._get_model_schema(prepared_model)
        )
        function_decl = types.FunctionDeclaration(
            name=gemini_utils._get_model_name(prepared_model),
            description=getattr(prepared_model, "__doc__", None),
            parameters=schema,
        )

        system_instruction = self._extract_system_instruction(new_kwargs)

        # Move OpenAI-style params to generation_config for conversion
        generation_config_dict = new_kwargs.pop("generation_config", {})
        for key in (
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
        ):
            if key in new_kwargs:
                generation_config_dict[key] = new_kwargs.pop(key)

        base_config = {
            "system_instruction": system_instruction,
            "tools": [types.Tool(function_declarations=[function_decl])],
            "tool_config": types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[
                        gemini_utils._get_model_name(prepared_model)
                    ],
                ),
            ),
        }
        # Temporarily put generation_config back for update_genai_kwargs to process
        new_kwargs["generation_config"] = generation_config_dict
        generation_config = gemini_utils.update_genai_kwargs(new_kwargs, base_config)
        new_kwargs.pop("generation_config", None)  # Remove it after processing
        new_kwargs["config"] = types.GenerateContentConfig(**generation_config)
        new_kwargs = self._convert_messages_to_contents(new_kwargs, autodetect_images)
        new_kwargs = self._cleanup_provider_kwargs(new_kwargs)
        return prepared_model, new_kwargs

    def handle_reask(
        self,
        *,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
        failed_attempts: list[Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        return gemini_utils.reask_genai_tools(
            kwargs.copy(),
            response,
            exception,
        )


@register_mode_handler(Provider.GENAI, Mode.JSON)
class GenAIStructuredOutputsHandler(GenAIHandlerBase):
    """Mode handler for GenAI structured outputs / JSON schema."""

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        from google.genai import types

        new_kwargs = self._clone_kwargs(kwargs)
        autodetect_images = self._pop_autodetect_images(new_kwargs)
        stream = bool(new_kwargs.get("stream", False))

        prepared_model = prepare_response_model(response_model)
        if prepared_model is None:
            return None, self._prepare_without_response_model(
                new_kwargs, autodetect_images
            )

        prepared_model = self._wrap_streaming_model(prepared_model, stream)
        # Validate schema for unsupported union types
        gemini_utils.map_to_gemini_function_schema(
            gemini_utils._get_model_schema(prepared_model)
        )

        system_instruction = self._extract_system_instruction(new_kwargs)

        # Move OpenAI-style params to generation_config for conversion
        generation_config_dict = new_kwargs.pop("generation_config", {})
        for key in (
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
        ):
            if key in new_kwargs:
                generation_config_dict[key] = new_kwargs.pop(key)

        base_config = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": prepared_model,
        }
        # Temporarily put generation_config back for update_genai_kwargs to process
        new_kwargs["generation_config"] = generation_config_dict
        generation_config = gemini_utils.update_genai_kwargs(new_kwargs, base_config)
        new_kwargs.pop("generation_config", None)  # Remove it after processing
        new_kwargs["config"] = types.GenerateContentConfig(**generation_config)
        new_kwargs = self._convert_messages_to_contents(new_kwargs, autodetect_images)
        new_kwargs = self._cleanup_provider_kwargs(new_kwargs)
        return prepared_model, new_kwargs

    def handle_reask(
        self,
        *,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
        failed_attempts: list[Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        return gemini_utils.reask_genai_structured_outputs(
            kwargs.copy(),
            response,
            exception,
        )
