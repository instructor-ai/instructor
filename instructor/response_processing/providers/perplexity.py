"""Perplexity-specific response handlers."""

from typing import Any, TypeVar

from instructor.mode import Mode
from instructor.response_processing.base import BaseHandler
from instructor.response_processing.registry import handler_registry

T = TypeVar("T")


class PerplexityJSONHandler(BaseHandler):
    """Handler for Perplexity JSON mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Perplexity JSON mode."""
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"schema": response_model.model_json_schema()},
        }
        return response_model, kwargs


def register_perplexity_handlers() -> None:
    """Register all Perplexity handlers."""
    handler_registry.register(Mode.PERPLEXITY_JSON, PerplexityJSONHandler())
