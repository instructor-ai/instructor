"""
OpenAI provider implementation for Instructor.
"""

from typing import Any, Union
from collections.abc import Iterator

try:
    from openai import OpenAI, AsyncOpenAI
    import importlib.util

    HAS_OPENAI = importlib.util.find_spec("openai") is not None
except ImportError:
    HAS_OPENAI = False

from pydantic import BaseModel

from ..mode import Mode
from ..client import Instructor, AsyncInstructor
from ..patch import openai_patch, openai_async_patch
from . import register_provider
from .base import ProviderBase


@register_provider("openai")
class OpenAIProvider(ProviderBase):
    """
    OpenAI provider implementation for Instructor.

    This provider supports:
    - Synchronous and asynchronous clients
    - JSON mode and function calling
    - Streaming responses
    - Response validation and retrying
    """

    _supported_modes = [Mode.JSON, Mode.TOOLS, Mode.OPENAI_TOOLS, Mode.FUNCTIONS]

    @classmethod
    def create_client(cls, client: Any, **kwargs) -> Union[Instructor, AsyncInstructor]:
        """
        Create an Instructor client from an OpenAI client.

        Args:
            client: An instance of openai.OpenAI or openai.AsyncOpenAI
            **kwargs: Additional arguments to pass to the Instructor constructor

        Returns:
            An instance of Instructor or AsyncInstructor
        """
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package is required to use the OpenAI provider. "
                "Install it with `pip install openai`."
            )

        # Handle both sync and async clients
        if isinstance(client, AsyncOpenAI):
            return AsyncInstructor(
                client=client,
                mode=kwargs.pop("mode", Mode.TOOLS),
                patched_client=openai_async_patch(client),
                **kwargs,
            )
        elif isinstance(client, OpenAI):
            return Instructor(
                client=client,
                mode=kwargs.pop("mode", Mode.TOOLS),
                patched_client=openai_patch(client),
                **kwargs,
            )
        else:
            raise TypeError(
                f"Expected openai.OpenAI or openai.AsyncOpenAI, got {type(client)}"
            )

    def create(
        self,
        response_model: type[BaseModel],
        messages: list[dict[str, Any]],
        mode: Mode,
        **kwargs,
    ) -> BaseModel:
        """
        Create structured output from messages.

        Args:
            response_model: Pydantic model to validate response
            messages: List of chat messages
            mode: Mode to use (JSON, TOOLS, FUNCTIONS)
            **kwargs: Additional arguments to pass to client.chat.completions.create

        Returns:
            Validated instance of response_model
        """
        self.validate_mode(mode)

        # This method exists to satisfy the interface
        # In a real implementation, all parameters would be used
        _ = response_model, messages  # Mark as used

        if mode == Mode.JSON:
            kwargs["response_format"] = {"type": "json_object"}
        elif mode in (Mode.TOOLS, Mode.OPENAI_TOOLS, Mode.FUNCTIONS):
            # Handle function calling configuration
            pass

        # Implementation would call the OpenAI client and process response
        # This is a placeholder for the actual implementation
        raise NotImplementedError("This is a test implementation")

    def create_stream(
        self,
        response_model: type[BaseModel],
        messages: list[dict[str, Any]],
        mode: Mode,
        **kwargs,
    ) -> Iterator[BaseModel]:
        """
        Stream structured output from messages.

        Args:
            response_model: Pydantic model to validate response
            messages: List of chat messages
            mode: Mode to use (JSON, TOOLS, FUNCTIONS)
            **kwargs: Additional arguments to pass to client.chat.completions.create

        Yields:
            Validated instances of response_model as they become available
        """
        self.validate_mode(mode)

        # This method exists to satisfy the interface
        # In a real implementation, all parameters would be used
        _ = response_model, messages, kwargs  # Mark as used

        # Implementation would stream from the OpenAI client
        # This is a placeholder for the actual implementation
        raise NotImplementedError("This is a test implementation")
