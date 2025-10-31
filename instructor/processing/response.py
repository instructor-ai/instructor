"""
This module serves as the central dispatcher for processing responses from various LLM providers
(OpenAI, Anthropic, Google, Cohere, etc.) and transforming them into structured Pydantic models.
It handles different response formats, streaming responses, validation, and error recovery.

The module supports 40+ different modes across providers, each with specific handling logic
for request formatting and response parsing. It also provides retry mechanisms (reask) for
handling validation errors gracefully.

Key Components:
    - Response processing functions for sync/async operations
    - Mode-based response model handlers for different providers
    - Error recovery and retry logic for validation failures
    - Support for streaming, partial, parallel, and iterable response models

Example:
    ```python
    from instructor.process_response import process_response
    from ..mode import Mode
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    # Process an OpenAI response
    processed = process_response(
        response=openai_response,
        response_model=User,
        mode=Mode.TOOLS,
        stream=False
    )
    ```
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncGenerator
from typing import Any, TypeVar, TYPE_CHECKING, cast

from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from typing_extensions import ParamSpec

from instructor.core.exceptions import InstructorError

from ..dsl.iterable import IterableBase
from ..dsl.parallel import ParallelBase
from ..dsl.partial import PartialBase
from ..dsl.simple_type import AdapterBase

if TYPE_CHECKING:
    from .function_calls import OpenAISchema
from ..mode import Mode
from .adapters import PROVIDER_ADAPTERS, ProviderAdapterContext
from .multimodal import convert_messages
from ..utils.core import prepare_response_model

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


async def extract_payload_async(
    response: ChatCompletion,
    response_model: type[T_Model | OpenAISchema | BaseModel] | None,
    *,
    stream: bool,
    validation_context: dict[str, Any] | None,
    strict: bool | None,
    mode: Mode,
) -> tuple[Any, bool]:
    """Collect the model payload for asynchronous response handling."""

    if response_model is None:
        return response, True

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        tasks: list[Any] = []
        async for task in response_model.from_streaming_response_async(  # type: ignore[arg-type]
            cast(AsyncGenerator[Any, None], response),
            mode=mode,
        ):
            tasks.append(task)
        return tasks, True

    payload = response_model.from_response(  # type: ignore[union-attr]
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )
    return payload, False


def extract_payload(
    response: Any,
    response_model: type[T_Model | OpenAISchema | BaseModel] | None,
    *,
    stream: bool,
    validation_context: dict[str, Any] | None,
    strict: bool | None,
    mode: Mode,
) -> tuple[Any, bool]:
    """Collect the model payload for synchronous response handling."""

    if response_model is None:
        return response, True

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        tasks = list(
            response_model.from_streaming_response(  # type: ignore[arg-type]
                response,
                mode=mode,
            )
        )
        return tasks, True

    payload = response_model.from_response(  # type: ignore[union-attr]
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )
    return payload, False


def coerce_to_model(
    payload: Any,
    *,
    response: Any,
    response_model: type[T_Model | OpenAISchema | BaseModel] | None,
) -> Any:
    """Normalize provider payloads to the expected return shape."""

    if response_model is None:
        return payload

    if isinstance(payload, list):
        return payload

    if isinstance(payload, IterableBase):
        return [task for task in payload.tasks]

    if isinstance(response_model, ParallelBase):
        return payload

    if isinstance(payload, AdapterBase):
        return payload.content

    setattr(payload, "_raw_response", response)
    return payload


async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: type[T_Model | OpenAISchema | BaseModel] | None,
    stream: bool = False,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | ChatCompletion:
    """Asynchronously process and transform LLM responses into structured models."""

    logger.debug("Instructor Raw Response: %s", response)
    payload, is_final = await extract_payload_async(
        response,
        response_model,
        stream=stream,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    if is_final:
        return payload  # type: ignore[return-value]

    return coerce_to_model(payload, response=response, response_model=response_model)


def process_response(
    response: T_Model,
    *,
    response_model: type[OpenAISchema | BaseModel] | None = None,
    stream: bool,
    validation_context: dict[str, Any] | None = None,
    strict=None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | list[T_Model] | None:
    """Process and transform LLM responses into structured models (synchronous)."""

    logger.debug("Instructor Raw Response: %s", response)

    payload, is_final = extract_payload(
        response,
        response_model,
        stream=stream,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    if is_final:
        return payload  # type: ignore[return-value]

    return coerce_to_model(payload, response=response, response_model=response_model)


def is_typed_dict(cls) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )


def apply_mode_adapter(
    response_model: type[T] | None,
    *,
    mode: Mode,
    kwargs: dict[str, Any],
    autodetect_images: bool,
) -> tuple[type[T] | None, dict[str, Any], "ProviderAdapter"]:
    """Apply the provider adapter to prepare request kwargs and response model."""

    adapter = PROVIDER_ADAPTERS.get(mode)
    if adapter is None:
        raise ValueError(f"Invalid patch mode: {mode}")

    prepared_model = response_model
    if prepared_model is not None and adapter.prepare_model:
        prepared_model = prepare_response_model(prepared_model)

    context = ProviderAdapterContext(
        mode=mode,
        autodetect_images=autodetect_images,
    )

    prepared_model, prepared_kwargs = adapter.prepare_request(  # type: ignore[arg-type]
        prepared_model,
        kwargs,
        context,
    )
    return prepared_model, prepared_kwargs, adapter


def handle_response_model(
    response_model: type[T] | None, mode: Mode = Mode.TOOLS, **kwargs: Any
) -> tuple[type[T] | None, dict[str, Any]]:
    """Prepare provider-specific arguments and response model."""

    new_kwargs = kwargs.copy()
    autodetect_images = new_kwargs.pop("autodetect_images", False)

    prepared_model, new_kwargs, adapter = apply_mode_adapter(
        response_model,
        mode=mode,
        kwargs=new_kwargs,
        autodetect_images=autodetect_images,
    )

    if adapter.convert_messages and "messages" in new_kwargs:
        new_kwargs["messages"] = convert_messages(
            new_kwargs["messages"],
            mode,
            autodetect_images=autodetect_images,
        )

    logger.debug(
        "Instructor Request: %s, response_model=%s, kwargs=%s",
        getattr(mode, "value", str(mode)),
        (
            prepared_model.__name__
            if prepared_model is not None and hasattr(prepared_model, "__name__")
            else str(prepared_model)
        ),
        new_kwargs,
    )
    return prepared_model, new_kwargs


def perform_reask(
    kwargs: dict[str, Any],
    mode: Mode,
    response: Any,
    exception: Exception,
    failed_attempts: list[Any] | None = None,
) -> dict[str, Any]:
    """Dispatch to the provider-specific reask handler."""

    adapter = PROVIDER_ADAPTERS.get(mode)
    if adapter is None:
        raise ValueError(f"Invalid patch mode: {mode}")

    kwargs_copy = kwargs.copy()
    enriched_exception = InstructorError.from_exception(
        exception, failed_attempts=failed_attempts
    )
    return adapter.reask(kwargs_copy, response, enriched_exception)
