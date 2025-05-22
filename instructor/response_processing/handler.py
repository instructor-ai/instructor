"""Main response model handler using the registry pattern."""

import logging
from typing import Any, Optional, TypeVar, Union

from instructor.dsl.parallel import (
    ParallelModel,
    VertexAIParallelBase,
    handle_parallel_model,
)
from instructor.exceptions import ConfigurationError
from instructor.mode import Mode
from instructor.multimodal import extract_genai_multimodal_content
from instructor.response_processing.core import prepare_response_model
from instructor.response_processing.messages import MessageHandler
from instructor.response_processing.providers import register_all_handlers
from instructor.response_processing.registry import handler_registry

logger = logging.getLogger("instructor")

T = TypeVar("T")

# Register all handlers on module import
register_all_handlers()


def handle_response_model(
    response_model: Optional[type[T]], mode: Mode = Mode.TOOLS, **kwargs: Any
) -> tuple[Optional[Union[type[T], VertexAIParallelBase]], dict[str, Any]]:
    """
    Handles the response model based on the specified mode and prepares the kwargs for the API call.

    Args:
        response_model: The response model to be used for parsing the API response.
        mode: The mode to use for handling the response model. Defaults to Mode.TOOLS.
        **kwargs: Additional keyword arguments to be passed to the API call.

    Returns:
        Tuple containing the processed response model and the updated kwargs.

    This function prepares the response model and modifies the kwargs based on the specified mode.
    It handles various modes like TOOLS, JSON, FUNCTIONS, etc., and applies the appropriate
    transformations to the response model and kwargs.
    """
    new_kwargs = kwargs.copy()
    autodetect_images = new_kwargs.pop("autodetect_images", False)

    if response_model is None:
        if mode in {Mode.COHERE_JSON_SCHEMA, Mode.COHERE_TOOLS}:
            # This is cause cohere uses 'message' and 'chat_history' instead of 'messages'
            return None, MessageHandler.prepare_cohere_messages(new_kwargs)

        # Handle images without a response model
        new_kwargs = MessageHandler.process_messages(
            new_kwargs, mode, autodetect_images
        )
        return None, new_kwargs

    # Handle special parallel tools modes
    if mode == Mode.PARALLEL_TOOLS:
        if new_kwargs.get("stream", False):
            raise ConfigurationError(
                "stream=True is not supported when using PARALLEL_TOOLS mode"
            )
        new_kwargs["tools"] = handle_parallel_model(response_model)
        new_kwargs["tool_choice"] = "auto"
        return ParallelModel(typehint=response_model), new_kwargs

    if mode == Mode.VERTEXAI_PARALLEL_TOOLS:
        # This is handled by the VertexAI provider
        pass
    else:
        # Prepare the response model for standard modes
        response_model = prepare_response_model(response_model)

    # Use the registry to handle the mode
    if not handler_registry.is_registered(mode):
        raise ValueError(f"Invalid patch mode: {mode}")

    response_model, new_kwargs = handler_registry.handle(
        mode, response_model, new_kwargs
    )

    # Process messages if present
    if "messages" in new_kwargs:
        new_kwargs["messages"] = MessageHandler.process_messages(
            new_kwargs, mode, autodetect_images
        )["messages"]

    # Special handling for GenAI modes
    if mode in {Mode.GENAI_TOOLS, Mode.GENAI_STRUCTURED_OUTPUTS}:
        new_kwargs["contents"] = extract_genai_multimodal_content(
            new_kwargs["contents"], autodetect_images
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
