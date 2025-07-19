# type: ignore[all]
from __future__ import annotations

import inspect
import logging
from typing import Any, TypeVar

from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from typing_extensions import ParamSpec

from instructor.dsl.iterable import IterableBase
from instructor.dsl.parallel import (
    ParallelBase,
    VertexAIParallelBase,
)
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import (
    AdapterBase,
)
from instructor.function_calls import OpenAISchema
from instructor.mode import Mode
from instructor.multimodal import convert_messages
from instructor.utils.anthropic import (
    handle_anthropic_tools,
    handle_anthropic_json,
    handle_anthropic_reasoning_tools,
    handle_anthropic_parallel_tools,
)
from instructor.utils.core import prepare_response_model
from instructor.utils.google import (
    handle_gemini_json,
    handle_gemini_tools,
    handle_genai_structured_outputs,
    handle_genai_tools,
    handle_vertexai_parallel_tools,
    handle_vertexai_tools,
    handle_vertexai_json,
)
from instructor.utils.openai import (
    handle_parallel_tools,
    handle_functions,
    handle_tools_strict,
    handle_tools,
    handle_responses_tools,
    handle_responses_tools_with_inbuilt_tools,
    handle_json_o1,
    handle_json_modes,
    handle_openrouter_structured_outputs,
)
from instructor.utils.cohere import (
    handle_cohere_json_schema,
    handle_cohere_tools,
)
from instructor.utils.mistral import (
    handle_mistral_tools,
    handle_mistral_structured_outputs,
)
from instructor.utils.bedrock import (
    handle_bedrock_json,
    handle_bedrock_tools,
)
from instructor.utils.fireworks import (
    handle_fireworks_tools,
    handle_fireworks_json,
)
from instructor.utils.cerebras import (
    handle_cerebras_tools,
    handle_cerebras_json,
)
from instructor.utils.writer import (
    handle_writer_tools,
    handle_writer_json,
)
from instructor.utils.perplexity import handle_perplexity_json
from instructor.utils.xai import (
    handle_xai_json,
    handle_xai_tools,
)

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: type[T_Model | OpenAISchema | BaseModel] | None,
    stream: bool = False,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | ChatCompletion:
    """
    Asynchronously processes the response from the OpenAI API.

    Args:
        response (ChatCompletion): The raw response from the OpenAI API.
        response_model (type[T_Model | OpenAISchema | BaseModel] | None): The expected model type for the response.
        stream (bool): Whether the response is streamed.
        validation_context (dict[str, Any] | None): Additional context for validation.
        strict (bool | None): Whether to apply strict validation.
        mode (Mode): The processing mode to use.

    Returns:
        T_Model | ChatCompletion: The processed response, either as the specified model type or the raw ChatCompletion.

    This function handles various response types, including streaming responses and different model bases.
    It applies the appropriate processing based on the response_model and mode provided.
    """

    logger.debug(
        f"Instructor Raw Response: {response}",
    )
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

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        logger.debug(f"Returning takes from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug(f"Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug(f"Returning model from AdapterBase")
        return model.content

    model._raw_response = response
    return model


def process_response(
    response: T_Model,
    *,
    response_model: type[OpenAISchema | BaseModel] | None = None,
    stream: bool,
    validation_context: dict[str, Any] | None = None,
    strict=None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | list[T_Model] | VertexAIParallelBase | None:
    """
    Process the response from the API call and convert it to the specified response model.

    Args:
        response (T_Model): The raw response from the API call.
        response_model (type[OpenAISchema | BaseModel] | None): The model to convert the response to.
        stream (bool): Whether the response is a streaming response.
        validation_context (dict[str, Any] | None): Additional context for validation.
        strict (bool | None): Whether to use strict validation.
        mode (Mode): The mode used for processing the response.

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
    logger.debug(
        f"Instructor Raw Response: {response}",
    )

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

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        logger.debug(f"Returning takes from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug(f"Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug(f"Returning model from AdapterBase")
        return model.content

    model._raw_response = response

    return model


def is_typed_dict(cls) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )


def handle_response_model(
    response_model: type[T] | None, mode: Mode = Mode.TOOLS, **kwargs: Any
) -> tuple[type[T] | VertexAIParallelBase | None, dict[str, Any]]:
    """
    Handles the response model based on the specified mode and prepares the kwargs for the API call.

    Args:
        response_model (type[T] | None): The response model to be used for parsing the API response.
        mode (Mode): The mode to use for handling the response model. Defaults to Mode.TOOLS.
        **kwargs: Additional keyword arguments to be passed to the API call.

    Returns:
        tuple[type[T] | None, dict[str, Any]]: A tuple containing the processed response model and the updated kwargs.

    This function prepares the response model and modifies the kwargs based on the specified mode.
    It handles various modes like TOOLS, JSON, FUNCTIONS, etc., and applies the appropriate
    transformations to the response model and kwargs.
    """

    new_kwargs = kwargs.copy()
    # print(f"instructor.process_response.py: new_kwargs -> {new_kwargs}")
    # Extract autodetect_images for message conversion
    autodetect_images = new_kwargs.pop("autodetect_images", False)

    PARALLEL_MODES = {
        Mode.PARALLEL_TOOLS: handle_parallel_tools,
        Mode.VERTEXAI_PARALLEL_TOOLS: handle_vertexai_parallel_tools,
        Mode.ANTHROPIC_PARALLEL_TOOLS: handle_anthropic_parallel_tools,
    }

    if mode in PARALLEL_MODES:
        response_model, new_kwargs = PARALLEL_MODES[mode](response_model, new_kwargs)
        logger.debug(
            f"Instructor Request: {mode.value=}, {response_model=}, {new_kwargs=}",
            extra={
                "mode": mode.value,
                "response_model": (
                    response_model.__name__
                    if response_model is not None
                    and hasattr(response_model, "__name__")
                    else str(response_model)
                ),
                "new_kwargs": new_kwargs,
            },
        )
        return response_model, new_kwargs

    # Only prepare response_model if it's not None
    if response_model is not None:
        response_model = prepare_response_model(response_model)

    mode_handlers = {  # type: ignore
        Mode.FUNCTIONS: handle_functions,
        Mode.TOOLS_STRICT: handle_tools_strict,
        Mode.TOOLS: handle_tools,
        Mode.MISTRAL_TOOLS: handle_mistral_tools,
        Mode.MISTRAL_STRUCTURED_OUTPUTS: handle_mistral_structured_outputs,
        Mode.JSON_O1: handle_json_o1,
        Mode.JSON: lambda rm, nk: handle_json_modes(rm, nk, Mode.JSON),  # type: ignore
        Mode.MD_JSON: lambda rm, nk: handle_json_modes(rm, nk, Mode.MD_JSON),  # type: ignore
        Mode.JSON_SCHEMA: lambda rm, nk: handle_json_modes(rm, nk, Mode.JSON_SCHEMA),  # type: ignore
        Mode.ANTHROPIC_TOOLS: handle_anthropic_tools,
        Mode.ANTHROPIC_REASONING_TOOLS: handle_anthropic_reasoning_tools,
        Mode.ANTHROPIC_JSON: handle_anthropic_json,
        Mode.COHERE_JSON_SCHEMA: handle_cohere_json_schema,
        Mode.COHERE_TOOLS: handle_cohere_tools,
        Mode.GEMINI_JSON: handle_gemini_json,
        Mode.GEMINI_TOOLS: handle_gemini_tools,
        Mode.GENAI_TOOLS: lambda rm, nk: handle_genai_tools(rm, nk, autodetect_images),
        Mode.GENAI_STRUCTURED_OUTPUTS: lambda rm, nk: handle_genai_structured_outputs(
            rm, nk, autodetect_images
        ),
        Mode.VERTEXAI_TOOLS: handle_vertexai_tools,
        Mode.VERTEXAI_JSON: handle_vertexai_json,
        Mode.CEREBRAS_JSON: handle_cerebras_json,
        Mode.CEREBRAS_TOOLS: handle_cerebras_tools,
        Mode.FIREWORKS_JSON: handle_fireworks_json,
        Mode.FIREWORKS_TOOLS: handle_fireworks_tools,
        Mode.WRITER_TOOLS: handle_writer_tools,
        Mode.WRITER_JSON: handle_writer_json,
        Mode.BEDROCK_JSON: handle_bedrock_json,
        Mode.BEDROCK_TOOLS: handle_bedrock_tools,
        Mode.PERPLEXITY_JSON: handle_perplexity_json,
        Mode.OPENROUTER_STRUCTURED_OUTPUTS: handle_openrouter_structured_outputs,
        Mode.RESPONSES_TOOLS: handle_responses_tools,
        Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: handle_responses_tools_with_inbuilt_tools,
        Mode.XAI_JSON: handle_xai_json,
        Mode.XAI_TOOLS: handle_xai_tools,
    }

    if mode in mode_handlers:
        response_model, new_kwargs = mode_handlers[mode](response_model, new_kwargs)
    else:
        raise ValueError(f"Invalid patch mode: {mode}")

    # Handle message conversion for modes that don't already handle it
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
