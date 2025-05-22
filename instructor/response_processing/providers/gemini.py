"""Gemini-specific response handlers."""

import json
from textwrap import dedent
from typing import Any, TypeVar

from instructor.exceptions import ConfigurationError
from instructor.mode import Mode
from instructor.response_processing.base import BaseHandler
from instructor.response_processing.registry import handler_registry
from instructor.utils import update_gemini_kwargs

T = TypeVar("T")


class GeminiJSONHandler(BaseHandler):
    """Handler for Gemini JSON mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Gemini JSON mode."""
        if "model" in kwargs:
            raise ConfigurationError(
                "Gemini `model` must be set while patching the client, not passed as a parameter to the create method"
            )

        message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n\n

            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

        if kwargs["messages"][0]["role"] != "system":
            kwargs["messages"].insert(0, {"role": "system", "content": message})
        else:
            kwargs["messages"][0]["content"] += f"\n\n{message}"

        kwargs["generation_config"] = kwargs.get("generation_config", {}) | {
            "response_mime_type": "application/json"
        }

        kwargs = update_gemini_kwargs(kwargs)
        return response_model, kwargs


class GeminiToolsHandler(BaseHandler):
    """Handler for Gemini tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Gemini tools mode."""
        if "model" in kwargs:
            raise ConfigurationError(
                "Gemini `model` must be set while patching the client, not passed as a parameter to the create method"
            )

        kwargs["tools"] = [response_model.gemini_schema]
        kwargs["tool_config"] = {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": [response_model.__name__],
            },
        }

        kwargs = update_gemini_kwargs(kwargs)
        return response_model, kwargs


def register_gemini_handlers() -> None:
    """Register all Gemini handlers."""
    handler_registry.register(Mode.GEMINI_JSON, GeminiJSONHandler())
    handler_registry.register(Mode.GEMINI_TOOLS, GeminiToolsHandler())
