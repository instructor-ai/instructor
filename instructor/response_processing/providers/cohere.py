"""Cohere-specific response handlers."""

from typing import Any, TypeVar

from instructor.mode import Mode
from instructor.response_processing.base import BaseHandler
from instructor.response_processing.messages import MessageHandler
from instructor.response_processing.registry import handler_registry

T = TypeVar("T")


class CohereJSONSchemaHandler(BaseHandler):
    """Handler for Cohere JSON schema mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Cohere JSON schema mode."""
        kwargs["response_format"] = {
            "type": "json_object",
            "schema": response_model.model_json_schema(),
        }
        _, kwargs = MessageHandler.prepare_cohere_messages(kwargs)
        return response_model, kwargs


class CohereToolsHandler(BaseHandler):
    """Handler for Cohere tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Cohere tools mode."""
        _, kwargs = MessageHandler.prepare_cohere_messages(kwargs)

        instruction = f"""\
Extract a valid {response_model.__name__} object based on the chat history and the json schema below.
{response_model.model_json_schema()}
The JSON schema was obtained by running:
```python
schema = {response_model.__name__}.model_json_schema()
```

The output must be a valid JSON object that `{response_model.__name__}.model_validate_json()` can successfully parse.
"""
        kwargs["chat_history"] = [{"role": "user", "message": instruction}] + kwargs[
            "chat_history"
        ]
        return response_model, kwargs


def register_cohere_handlers() -> None:
    """Register all Cohere handlers."""
    handler_registry.register(Mode.COHERE_JSON_SCHEMA, CohereJSONSchemaHandler())
    handler_registry.register(Mode.COHERE_TOOLS, CohereToolsHandler())
