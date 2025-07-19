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
from typing import Any, TypeVar, TYPE_CHECKING

from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from typing_extensions import ParamSpec

from ..dsl.iterable import IterableBase
from ..dsl.parallel import ParallelBase
from ..dsl.partial import PartialBase
from ..dsl.simple_type import AdapterBase

if TYPE_CHECKING:
    from .function_calls import OpenAISchema
from ..mode import Mode
from .multimodal import convert_messages
from ..utils.core import prepare_response_model

# Anthropic utils
from ..providers.anthropic.utils import (
    handle_anthropic_json,
    handle_anthropic_parallel_tools,
    handle_anthropic_reasoning_tools,
    handle_anthropic_tools,
    reask_anthropic_json,
    reask_anthropic_tools,
)

# Bedrock utils
from ..providers.bedrock.utils import (
    handle_bedrock_json,
    handle_bedrock_tools,
    reask_bedrock_json,
    reask_bedrock_tools,
)

# Cerebras utils
from ..providers.cerebras.utils import (
    handle_cerebras_json,
    handle_cerebras_tools,
    reask_cerebras_tools,
)

# Cohere utils
from ..providers.cohere.utils import (
    handle_cohere_json_schema,
    handle_cohere_tools,
    reask_cohere_tools,
)

# Fireworks utils
from ..providers.fireworks.utils import (
    handle_fireworks_json,
    handle_fireworks_tools,
    reask_fireworks_json,
    reask_fireworks_tools,
)

# Google/Gemini/VertexAI utils
from ..providers.gemini.utils import (
    handle_gemini_json,
    handle_gemini_tools,
    handle_genai_structured_outputs,
    handle_genai_tools,
    handle_vertexai_json,
    handle_vertexai_parallel_tools,
    handle_vertexai_tools,
    reask_gemini_json,
    reask_gemini_tools,
    reask_genai_structured_outputs,
    reask_genai_tools,
    reask_vertexai_json,
    reask_vertexai_tools,
)

# Mistral utils
from ..providers.mistral.utils import (
    handle_mistral_structured_outputs,
    handle_mistral_tools,
    reask_mistral_structured_outputs,
    reask_mistral_tools,
)

# OpenAI utils
from ..providers.openai.utils import (
    handle_functions,
    handle_json_modes,
    handle_json_o1,
    handle_openrouter_structured_outputs,
    handle_parallel_tools,
    handle_responses_tools,
    handle_responses_tools_with_inbuilt_tools,
    handle_tools,
    handle_tools_strict,
    reask_default,
    reask_md_json,
    reask_responses_tools,
    reask_tools,
)

# Perplexity utils
from ..providers.perplexity.utils import (
    handle_perplexity_json,
    reask_perplexity_json,
)

# Writer utils
from ..providers.writer.utils import (
    handle_writer_json,
    handle_writer_tools,
    reask_writer_json,
    reask_writer_tools,
)

# XAI utils
from ..providers.xai.utils import (
    handle_xai_json,
    handle_xai_tools,
    reask_xai_json,
    reask_xai_tools,
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
    """Asynchronously process and transform LLM responses into structured models.

    This function is the async entry point for converting raw LLM responses into validated
    Pydantic models. It handles various response formats from different providers and
    supports special response types like streaming, partial objects, and parallel tool calls.

    Args:
        response (ChatCompletion or Similar API Response): The raw response from the LLM API. Despite the type hint,
            this can be responses from any supported provider (OpenAI, Anthropic, Google, etc.)
        response_model (type[T_Model | BaseModel] | None): The target Pydantic
            model to parse the response into. If None, returns the raw response unchanged.
            Can also be special DSL types like ParallelBase for parallel tool calls, or IterableBase and PartialBase for streaming.
        stream (bool): Whether this is a streaming response. Required for proper handling
            of IterableBase and PartialBase models. Defaults to False.
        validation_context (dict[str, Any] | None): Additional context passed to Pydantic
            validators during model validation. Useful for dynamic validation logic. The context
            is also used to format templated responses. Defaults to None.
        strict (bool | None): Whether to enforce strict JSON parsing. When True, the response
            must exactly match the model schema. When False, allows minor deviations.
        mode (Mode): The provider/format mode that determines how to parse the response.
            Examples: Mode.TOOLS (OpenAI), Mode.ANTHROPIC_JSON, Mode.GEMINI_TOOLS.
            Defaults to Mode.TOOLS.

    Returns:
        T_Model | ChatCompletion: The processed response. Return type depends on inputs:
            - If response_model is None: returns raw response unchanged
            - If response_model is IterableBase with stream=True: returns list of models
            - If response_model is AdapterBase: returns the adapted content
            - Otherwise: returns instance of response_model with _raw_response attached

    Raises:
        ValidationError: If the response doesn't match the expected model schema
        IncompleteOutputException: If the response was truncated due to token limits
        ValueError: If an invalid mode is specified

    Note:
        The function automatically detects special response model types (Iterable, Partial,
        Parallel, Adapter) and applies appropriate processing logic for each.
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
) -> T_Model | list[T_Model] | None:
    """Process and transform LLM responses into structured models (synchronous).

    This is the main entry point for converting raw LLM responses into validated Pydantic
    models. It acts as a dispatcher that handles various response formats from 40+ different
    provider modes and transforms them according to the specified response model type.

    Args:
        response (T_Model): The raw response from the LLM API. The actual type varies by
            provider (ChatCompletion for OpenAI, Message for Anthropic, etc.)
        response_model (type[OpenAISchema | BaseModel] | None): The target Pydantic model
            class to parse the response into. Special DSL types supported:
            - IterableBase: For streaming multiple objects from a single response
            - PartialBase: For incomplete/streaming partial objects
            - ParallelBase: For parallel tool/function calls
            - AdapterBase: For simple type adaptations (e.g., str, int)
            If None, returns the raw response unchanged.
        stream (bool): Whether this is a streaming response. Required to be True for
            proper handling of IterableBase and PartialBase models.
        validation_context (dict[str, Any] | None): Additional context passed to Pydantic
            validators. Useful for runtime validation logic based on external state.
        strict (bool | None): Controls JSON parsing strictness:
            - True: Enforce exact schema matching (no extra fields)
            - False/None: Allow minor deviations and extra fields
        mode (Mode): The provider/format mode that determines parsing strategy.
            Each mode corresponds to a specific provider and format combination:
            - Tool modes: TOOLS, ANTHROPIC_TOOLS, GEMINI_TOOLS, etc.
            - JSON modes: JSON, ANTHROPIC_JSON, VERTEXAI_JSON, etc.
            - Special modes: PARALLEL_TOOLS, MD_JSON, JSON_SCHEMA, etc.

    Returns:
        T_Model | list[T_Model] | None: The processed response:
            - If response_model is None: Original response unchanged
            - If IterableBase: List of extracted model instances
            - If ParallelBase: Special parallel response object
            - If AdapterBase: The adapted simple type (str, int, etc.)
            - Otherwise: Single instance of response_model with _raw_response attached

    Raises:
        ValidationError: Response doesn't match the expected model schema
        IncompleteOutputException: Response truncated due to token limits
        ValueError: Invalid mode specified or mode not supported
        JSONDecodeError: Malformed JSON in response (for JSON modes)

    Note:
        The function preserves the raw response by attaching it to the parsed model
        as `_raw_response`. This allows access to metadata like token usage, model
        info, and other provider-specific fields after parsing.
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
) -> tuple[type[T] | None, dict[str, Any]]:
    """
    Handles the response model based on the specified mode and prepares the kwargs for the API call.
    This really should be named 'prepare_create_kwargs' as its job is to map the openai create kwargs
    to the correct format for the API call based on the mode.

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


def handle_reask_kwargs(
    kwargs: dict[str, Any],
    mode: Mode,
    response: Any,
    exception: Exception,
) -> dict[str, Any]:
    """Handle validation errors by reformatting the request for retry (reask).

    When a response fails validation (e.g., missing required fields, wrong types),
    this function prepares a new request that includes information about the error.
    This allows the LLM to understand what went wrong and correct its response.

    The reask logic is provider-specific because each provider has different ways
    of handling function/tool calls and different message formats.

    Args:
        kwargs (dict[str, Any]): The original request parameters that resulted in
            a validation error. Includes messages, tools, temperature, etc.
        mode (Mode): The provider/format mode that determines which reask handler
            to use. Each mode has a specific strategy for formatting error feedback.
        response (Any): The raw response from the LLM that failed validation.
            Type varies by provider:
            - OpenAI: ChatCompletion with tool_calls
            - Anthropic: Message with tool_use blocks
            - Google: GenerateContentResponse with function calls
        exception (Exception): The validation error that occurred. Usually a
            Pydantic ValidationError with details about which fields failed.

    Returns:
        dict[str, Any]: Modified kwargs for the retry request, typically including:
            - Updated messages with error context
            - Same tool/function definitions
            - Preserved generation parameters
            - Provider-specific formatting

    Reask Strategies by Provider:
        Each provider has a specific strategy for handling retries:

        **JSON Modes:**
        - Adds assistant message with failed attempt
        - Adds user message with error details

        **Tool Calls:**
        - Preserves tool definitions
        - Formats the errors as tool calls responses

    Note:
        This function is typically called internally by the retry logic when
        max_retries > 1. It ensures that each retry attempt includes context
        about previous failures, helping the LLM learn from its mistakes.
    """
    # Create a shallow copy of kwargs to avoid modifying the original
    kwargs_copy = kwargs.copy()

    # Organized by provider (matching process_response.py structure)
    REASK_HANDLERS = {
        # OpenAI modes
        Mode.FUNCTIONS: reask_default,
        Mode.TOOLS_STRICT: reask_tools,
        Mode.TOOLS: reask_tools,
        Mode.JSON_O1: reask_default,
        Mode.JSON: reask_md_json,
        Mode.MD_JSON: reask_md_json,
        Mode.JSON_SCHEMA: reask_md_json,
        Mode.PARALLEL_TOOLS: reask_tools,
        Mode.RESPONSES_TOOLS: reask_responses_tools,
        Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: reask_responses_tools,
        # Mistral modes
        Mode.MISTRAL_TOOLS: reask_mistral_tools,
        Mode.MISTRAL_STRUCTURED_OUTPUTS: reask_mistral_structured_outputs,
        # Anthropic modes
        Mode.ANTHROPIC_TOOLS: reask_anthropic_tools,
        Mode.ANTHROPIC_REASONING_TOOLS: reask_anthropic_tools,
        Mode.ANTHROPIC_JSON: reask_anthropic_json,
        Mode.ANTHROPIC_PARALLEL_TOOLS: reask_anthropic_tools,
        # Cohere modes
        Mode.COHERE_TOOLS: reask_cohere_tools,
        Mode.COHERE_JSON_SCHEMA: reask_cohere_tools,
        # Gemini/Google modes
        Mode.GEMINI_TOOLS: reask_gemini_tools,
        Mode.GEMINI_JSON: reask_gemini_json,
        Mode.GENAI_TOOLS: reask_genai_tools,
        Mode.GENAI_STRUCTURED_OUTPUTS: reask_genai_structured_outputs,
        # VertexAI modes
        Mode.VERTEXAI_TOOLS: reask_vertexai_tools,
        Mode.VERTEXAI_JSON: reask_vertexai_json,
        Mode.VERTEXAI_PARALLEL_TOOLS: reask_vertexai_tools,
        # Cerebras modes
        Mode.CEREBRAS_TOOLS: reask_cerebras_tools,
        Mode.CEREBRAS_JSON: reask_default,
        # Fireworks modes
        Mode.FIREWORKS_TOOLS: reask_fireworks_tools,
        Mode.FIREWORKS_JSON: reask_fireworks_json,
        # Writer modes
        Mode.WRITER_TOOLS: reask_writer_tools,
        Mode.WRITER_JSON: reask_writer_json,
        # Bedrock modes
        Mode.BEDROCK_TOOLS: reask_bedrock_tools,
        Mode.BEDROCK_JSON: reask_bedrock_json,
        # Perplexity modes
        Mode.PERPLEXITY_JSON: reask_perplexity_json,
        # OpenRouter modes
        Mode.OPENROUTER_STRUCTURED_OUTPUTS: reask_default,
        # XAI modes
        Mode.XAI_JSON: reask_xai_json,
        Mode.XAI_TOOLS: reask_xai_tools,
    }

    if mode in REASK_HANDLERS:
        return REASK_HANDLERS[mode](kwargs_copy, response, exception)
    else:
        return reask_default(kwargs_copy, response, exception)
