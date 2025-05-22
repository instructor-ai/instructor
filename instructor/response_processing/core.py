"""Core response processing functionality."""

import inspect
import logging
from collections.abc import Iterable
from typing import Any, Optional, TypeVar, Union, get_args, get_origin

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, create_model

from instructor.dsl.iterable import IterableBase, IterableModel
from instructor.dsl.parallel import ParallelBase, VertexAIParallelBase
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import AdapterBase, ModelAdapter, is_simple_type
from instructor.function_calls import OpenAISchema, openai_schema
from instructor.mode import Mode

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T = TypeVar("T")


async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: Optional[type[Union[T_Model, OpenAISchema, BaseModel]]],
    stream: bool = False,
    validation_context: Optional[dict[str, Any]] = None,
    strict: Optional[bool] = None,
    mode: Mode = Mode.TOOLS,
) -> Union[T_Model, ChatCompletion]:
    """
    Asynchronously processes the response from the OpenAI API.

    Args:
        response: The raw response from the OpenAI API.
        response_model: The expected model type for the response.
        stream: Whether the response is streamed.
        validation_context: Additional context for validation.
        strict: Whether to apply strict validation.
        mode: The processing mode to use.

    Returns:
        The processed response, either as the specified model type or the raw ChatCompletion.

    This function handles various response types, including streaming responses and different model bases.
    It applies the appropriate processing based on the response_model and mode provided.
    """
    logger.debug(f"Instructor Raw Response: {response}")

    if response_model is None:
        return response

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

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # Process different model types
    if isinstance(model, IterableBase):
        logger.debug("Returning tasks from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug("Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug("Returning model from AdapterBase")
        return model.content

    model._raw_response = response
    return model


def process_response(
    response: T_Model,
    *,
    response_model: Optional[type[Union[OpenAISchema, BaseModel]]] = None,
    stream: bool,
    validation_context: Optional[dict[str, Any]] = None,
    strict=None,
    mode: Mode = Mode.TOOLS,
) -> Optional[Union[T_Model, list[T_Model], VertexAIParallelBase]]:
    """
    Process the response from the API call and convert it to the specified response model.

    Args:
        response: The raw response from the API call.
        response_model: The model to convert the response to.
        stream: Whether the response is a streaming response.
        validation_context: Additional context for validation.
        strict: Whether to use strict validation.
        mode: The mode used for processing the response.

    Returns:
        The processed response, which could be:
        - The raw response if no response_model is specified
        - An instance of the response_model
        - A list of tasks if the model is an IterableBase
        - The content of the model if it's an AdapterBase

    This function handles various types of responses and models, including streaming
    responses, iterable models, parallel models, and adapter models. It also attaches
    the raw response to the processed model when applicable.
    """
    logger.debug(f"Instructor Raw Response: {response}")

    if response_model is None:
        logger.debug("No response model, returning response as is")
        return response

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

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # Process different model types
    if isinstance(model, IterableBase):
        logger.debug("Returning tasks from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug("Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug("Returning model from AdapterBase")
        return model.content

    model._raw_response = response
    return model


def is_typed_dict(cls) -> bool:
    """Check if a class is a TypedDict."""
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )


def prepare_response_model(response_model: Optional[type[T]]) -> Optional[type[T]]:
    """
    Prepares the response model for use in the API call.

    This function performs several transformations on the input response_model:
    1. If the response_model is None, it returns None.
    2. If it's a simple type, it wraps it in a ModelAdapter.
    3. If it's a TypedDict, it converts it to a Pydantic BaseModel.
    4. If it's an Iterable, it wraps the element type in an IterableModel.
    5. If it's not already a subclass of OpenAISchema, it applies the openai_schema decorator.

    Args:
        response_model: The input response model to be prepared.

    Returns:
        The prepared response model, or None if the input was None.
    """
    if response_model is None:
        return None

    if is_simple_type(response_model):
        response_model = ModelAdapter[response_model]

    if is_typed_dict(response_model):
        response_model: BaseModel = create_model(
            response_model.__name__,
            **{k: (v, ...) for k, v in response_model.__annotations__.items()},
        )

    if get_origin(response_model) is Iterable:
        iterable_element_class = get_args(response_model)[0]
        response_model = IterableModel(iterable_element_class)

    if not issubclass(response_model, OpenAISchema):
        response_model = openai_schema(response_model)  # type: ignore

    return response_model
