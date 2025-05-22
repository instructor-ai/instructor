"""Refactored response processing using provider pattern."""

from __future__ import annotations

import inspect
import logging
from typing import Any, TypeVar

from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from instructor.dsl.iterable import IterableBase
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import ModelAdapter, is_simple_type
from instructor.function_calls import OpenAISchema
from instructor.mode import Mode
from instructor.multimodal import convert_messages
from instructor.providers import ProviderRegistry
from instructor.exceptions import ProviderNotFoundError, InvalidModeError

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)


def prepare_response_model(response_model: type[T_Model]) -> type[T_Model]:
    """Wrap response model in ModelAdapter if it's a simple type."""
    if is_simple_type(response_model):
        return ModelAdapter[response_model]  # type: ignore
    return response_model


async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: type[T_Model | OpenAISchema | BaseModel] | None,
    stream: bool = False,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
    provider: str | None = None,
) -> T_Model | ChatCompletion:
    """
    Asynchronously processes the response using the provider pattern.

    Args:
        response: The raw response from the API
        response_model: The expected model type for the response
        stream: Whether the response is streamed
        validation_context: Additional context for validation
        strict: Whether to apply strict validation
        mode: The processing mode to use
        provider: The provider name (if None, will be detected from mode)

    Returns:
        The processed response, either as the specified model type or raw response
    """
    logger.debug(f"Instructor Raw Response: {response}")

    if response_model is None:
        return response

    # Handle streaming iterables and partials
    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = await response_model.from_streaming_response_async(
            response,
            mode=mode,
        )
        return model

    # Get the appropriate provider
    if provider:
        provider_instance = ProviderRegistry.get_provider(provider)
    else:
        provider_instance = ProviderRegistry.get_provider_for_mode(mode)
        if not provider_instance:
            raise ProviderNotFoundError(f"No provider found for mode: {mode}")

    # Validate mode support
    if not provider_instance.supports_mode(mode):
        raise InvalidModeError(
            f"Provider '{provider_instance.name}' does not support mode '{mode}'"
        )

    # Process response using provider
    return await provider_instance.process_response_async(
        response,
        response_model,
        mode,
        validation_context=validation_context,
        strict=strict,
    )


def process_response(
    response: ChatCompletion,
    *,
    response_model: type[T_Model | OpenAISchema | BaseModel] | None,
    stream: bool = False,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
    provider: str | None = None,
) -> T_Model | ChatCompletion:
    """
    Synchronously processes the response using the provider pattern.

    Args:
        response: The raw response from the API
        response_model: The expected model type for the response
        stream: Whether the response is streamed
        validation_context: Additional context for validation
        strict: Whether to apply strict validation
        mode: The processing mode to use
        provider: The provider name (if None, will be detected from mode)

    Returns:
        The processed response, either as the specified model type or raw response
    """
    logger.debug(f"Instructor Raw Response: {response}")

    if response_model is None:
        return response

    # Handle streaming iterables and partials
    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = response_model.from_streaming_response(
            response,
            mode=mode,
        )
        return model

    # Get the appropriate provider
    if provider:
        provider_instance = ProviderRegistry.get_provider(provider)
    else:
        provider_instance = ProviderRegistry.get_provider_for_mode(mode)
        if not provider_instance:
            raise ProviderNotFoundError(f"No provider found for mode: {mode}")

    # Validate mode support
    if not provider_instance.supports_mode(mode):
        raise InvalidModeError(
            f"Provider '{provider_instance.name}' does not support mode '{mode}'"
        )

    # Process response using provider
    return provider_instance.process_response(
        response,
        response_model,
        mode,
        validation_context=validation_context,
        strict=strict,
    )


def handle_response_model(
    response_model: type[T_Model] | None,
    mode: Mode = Mode.TOOLS,
    provider: str | None = None,
    **kwargs: Any,
) -> tuple[type[T_Model] | None, dict[str, Any]]:
    """
    Handle response model preparation using the provider pattern.

    Args:
        response_model: The response model to prepare
        mode: The mode to use
        provider: The provider name (if None, will be detected from mode)
        **kwargs: Additional arguments to pass to the provider

    Returns:
        Tuple of prepared response model and updated kwargs
    """
    new_kwargs = kwargs.copy()
    autodetect_images = new_kwargs.pop("autodetect_images", False)

    # Handle case with no response model
    if response_model is None:
        if "messages" in new_kwargs:
            messages = convert_messages(
                new_kwargs["messages"],
                mode,
                autodetect_images=autodetect_images,
            )
            new_kwargs["messages"] = messages
        return None, new_kwargs

    # Get the appropriate provider
    if provider:
        provider_instance = ProviderRegistry.get_provider(provider)
    else:
        provider_instance = ProviderRegistry.get_provider_for_mode(mode)
        if not provider_instance:
            raise ProviderNotFoundError(f"No provider found for mode: {mode}")

    # Validate mode support
    if not provider_instance.supports_mode(mode):
        raise InvalidModeError(
            f"Provider '{provider_instance.name}' does not support mode '{mode}'"
        )

    # Prepare response model
    response_model = prepare_response_model(response_model)

    # Let provider prepare the request
    if hasattr(provider_instance, "prepare_request"):
        response_model, new_kwargs = provider_instance.prepare_request(
            response_model, new_kwargs, mode
        )

    # Convert messages if present
    if "messages" in new_kwargs:
        new_kwargs["messages"] = convert_messages(
            new_kwargs["messages"],
            mode,
            autodetect_images=autodetect_images,
        )

    logger.debug(
        f"Instructor Request: {mode.value=}, {response_model=}, {new_kwargs=}",
        extra={
            "mode": mode.value,
            "response_model": (
                response_model.__name__
                if response_model is not None and hasattr(response_model, "__name__")
                else str(response_model)
            ),
            "new_kwargs": new_kwargs,
        },
    )

    return response_model, new_kwargs
