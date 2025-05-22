"""VertexAI-specific response handlers."""

from typing import Any, TypeVar
from collections.abc import Iterable

from instructor.dsl.parallel import (
    VertexAIParallelBase,
    VertexAIParallelModel,
    get_types_array,
)
from instructor.exceptions import ConfigurationError
from instructor.mode import Mode
from instructor.response_processing.base import BaseHandler
from instructor.response_processing.registry import handler_registry

T = TypeVar("T")


class VertexAIParallelToolsHandler(BaseHandler):
    """Handler for VertexAI parallel tools mode."""

    def handle(
        self, response_model: type[Iterable[T]], kwargs: dict[str, Any]
    ) -> tuple[VertexAIParallelBase, dict[str, Any]]:
        """Handle VertexAI parallel tools mode."""
        if kwargs.get("stream", False):
            raise ConfigurationError(
                "stream=True is not supported when using VERTEXAI_PARALLEL_TOOLS mode"
            )

        from instructor.client_vertexai import vertexai_process_response

        # Extract concrete types before passing to vertexai_process_response
        model_types = list(get_types_array(response_model))
        contents, tools, tool_config = vertexai_process_response(kwargs, model_types)
        kwargs["contents"] = contents
        kwargs["tools"] = tools
        kwargs["tool_config"] = tool_config

        return VertexAIParallelModel(typehint=response_model), kwargs


class VertexAIToolsHandler(BaseHandler):
    """Handler for VertexAI tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle VertexAI tools mode."""
        from instructor.client_vertexai import vertexai_process_response

        contents, tools, tool_config = vertexai_process_response(kwargs, response_model)

        kwargs["contents"] = contents
        kwargs["tools"] = tools
        kwargs["tool_config"] = tool_config
        return response_model, kwargs


class VertexAIJSONHandler(BaseHandler):
    """Handler for VertexAI JSON mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle VertexAI JSON mode."""
        from instructor.client_vertexai import vertexai_process_json_response

        contents, generation_config = vertexai_process_json_response(
            kwargs, response_model
        )

        kwargs["contents"] = contents
        kwargs["generation_config"] = generation_config
        return response_model, kwargs


def register_vertexai_handlers() -> None:
    """Register all VertexAI handlers."""
    handler_registry.register(
        Mode.VERTEXAI_PARALLEL_TOOLS, VertexAIParallelToolsHandler()
    )
    handler_registry.register(Mode.VERTEXAI_TOOLS, VertexAIToolsHandler())
    handler_registry.register(Mode.VERTEXAI_JSON, VertexAIJSONHandler())
