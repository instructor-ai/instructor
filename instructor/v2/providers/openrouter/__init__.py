"""OpenRouter v2 provider handlers and client."""

from .client import from_openrouter
from .handlers import (
    OpenRouterJSONSchemaHandler,
    OpenRouterMDJSONHandler,
    OpenRouterParallelToolsHandler,
    OpenRouterToolsHandler,
)

__all__ = [
    "OpenRouterJSONSchemaHandler",
    "OpenRouterMDJSONHandler",
    "OpenRouterParallelToolsHandler",
    "OpenRouterToolsHandler",
    "from_openrouter",
]
