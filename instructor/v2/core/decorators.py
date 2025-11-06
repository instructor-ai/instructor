"""Decorator utilities for v2 mode registration."""

from instructor.v2.core.mode_types import ModeType, Provider
from instructor.v2.core.registry import mode_registry


def register_mode_handler(
    provider: Provider,
    mode_type: ModeType,
):
    """Decorator to register a mode handler class.

    The decorated class must implement RequestHandler, ReaskHandler,
    and ResponseParser protocols via prepare_request, handle_reask,
    and parse_response methods.

    Args:
        provider: Provider enum value
        mode_type: Mode type enum value

    Returns:
        Decorator function

    Example:
        >>> @register_mode_handler(Provider.ANTHROPIC, ModeType.TOOLS)
        ... class AnthropicToolsHandler:
        ...     def prepare_request(self, response_model, kwargs):
        ...         return response_model, kwargs
        ...     def handle_reask(self, kwargs, response, exception):
        ...         return kwargs
        ...     def parse_response(self, response, response_model, **kwargs):
        ...         return response_model.model_validate(response)
    """

    def decorator(handler_class: type) -> type:
        """Register the handler class."""
        # Instantiate the handler
        handler = handler_class()

        # Register with the registry
        mode_registry.register(
            provider=provider,
            mode_type=mode_type,
            request_handler=handler.prepare_request,
            reask_handler=handler.handle_reask,
            response_parser=handler.parse_response,
        )

        return handler_class

    return decorator
