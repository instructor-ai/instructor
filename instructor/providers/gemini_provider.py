"""Google Gemini provider implementation."""

from typing import Any, TypeVar, Union

from pydantic import BaseModel

from ..mode import Mode
from ..client import Instructor, AsyncInstructor
from ..patch import patch, apatch
from ..dsl.simple_type import ModelAdapter, is_simple_type
from ..utils import map_to_gemini_function_schema
from .base import BaseProvider
from .registry import ProviderRegistry

T_Model = TypeVar("T_Model", bound=BaseModel)


@ProviderRegistry.register("gemini")
class GeminiProvider(BaseProvider):
    """Provider implementation for Google Gemini."""

    @property
    def name(self) -> str:
        return "gemini"

    def get_supported_modes(self) -> set[Mode]:
        return {
            Mode.GEMINI_JSON,
            Mode.GEMINI_TOOLS,
        }

    def validate_response(self, response: Any, mode: Mode) -> None:
        """Validate Gemini response format."""
        # Gemini-specific validation
        pass

    def process_response(
        self, response: Any, response_model: type[T_Model], mode: Mode, **kwargs: Any
    ) -> T_Model:
        """Process Gemini response based on mode."""
        from ..process_response import process_response as legacy_process

        return legacy_process(
            response, response_model=response_model, mode=mode, **kwargs
        )

    async def process_response_async(
        self, response: Any, response_model: type[T_Model], mode: Mode, **kwargs: Any
    ) -> T_Model:
        """Process Gemini response asynchronously."""
        from ..process_response import process_response_async as legacy_process_async

        return await legacy_process_async(
            response, response_model=response_model, mode=mode, **kwargs
        )

    def create_instructor(
        self, client: Any, mode: Mode, **kwargs: Any
    ) -> Union[Instructor, AsyncInstructor]:
        """Create an instructor instance for Gemini."""
        create_fn = kwargs.pop("create", None)
        # Detect async client
        if hasattr(client, "generate_content_async"):
            return apatch(create=create_fn)(client, mode=mode)
        return patch(create=create_fn)(client, mode=mode)

    def prepare_request(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any], mode: Mode
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Prepare request parameters based on mode."""
        if mode == Mode.GEMINI_JSON:
            return self._handle_gemini_json(response_model, new_kwargs)
        elif mode == Mode.GEMINI_TOOLS:
            return self._handle_gemini_tools(response_model, new_kwargs)
        else:
            raise ValueError(f"Unsupported mode for Gemini provider: {mode}")

    def _handle_gemini_json(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any]
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Handle GEMINI_JSON mode."""
        import google.generativeai as genai

        if is_simple_type(response_model):
            response_model = ModelAdapter[response_model]  # type: ignore

        # Configure generation with JSON schema
        schema = response_model.model_json_schema()
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=schema,
        )

        # Update generation config
        if "generation_config" in new_kwargs:
            # Merge with existing config
            existing_config = new_kwargs["generation_config"]
            if hasattr(existing_config, "_pb"):
                # It's already a GenerationConfig object
                existing_config.response_mime_type = "application/json"
                existing_config.response_schema = schema
            else:
                # It's a dict
                existing_config["response_mime_type"] = "application/json"
                existing_config["response_schema"] = schema
        else:
            new_kwargs["generation_config"] = generation_config

        return response_model, new_kwargs

    def _handle_gemini_tools(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any]
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Handle GEMINI_TOOLS mode."""
        if is_simple_type(response_model):
            response_model = ModelAdapter[response_model]  # type: ignore

        # Get Gemini function schema
        gemini_function = map_to_gemini_function_schema(response_model.gemini_schema())
        new_kwargs["tools"] = [gemini_function]

        # Configure tool behavior
        from google.generativeai.types import content_types

        new_kwargs["tool_config"] = content_types.ToolConfig(
            function_calling_config=content_types.FunctionCallingConfig(
                mode=content_types.FunctionCallingConfig.Mode.ANY,
                allowed_function_names=[gemini_function.name],
            )
        )

        return response_model, new_kwargs
