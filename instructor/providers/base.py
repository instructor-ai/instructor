"""
Base provider interfaces for Instructor.

This module defines the base provider protocols and classes that all providers should implement.
"""

from abc import abstractmethod
from typing import Protocol, TypeVar, Generic, Any, Optional, Union, ClassVar
from collections.abc import Iterator
from pydantic import BaseModel

from ..mode import Mode
from ..client import Instructor, AsyncInstructor


# Type variable for response model
T = TypeVar("T", bound=BaseModel)


class ProviderProtocol(Protocol, Generic[T]):
    """Protocol defining what all providers must implement."""

    @property
    def supported_modes(self) -> list[Mode]:
        """Modes supported by this provider."""
        ...

    @classmethod
    def capabilities(cls) -> dict[str, bool]:
        """Provider capabilities (streaming, function_calling, etc.)"""
        ...

    @classmethod
    def create_client(
        cls, client: Any, provider_id: Optional[str] = None, **kwargs
    ) -> Union[Instructor, AsyncInstructor]:
        """Create an Instructor client for this provider."""
        ...

    def create(
        self,
        response_model: type[T],
        messages: list[dict[str, Any]],
        mode: Mode,
        **kwargs,
    ) -> T:
        """Create structured output from messages."""
        ...

    def create_stream(
        self,
        response_model: type[T],
        messages: list[dict[str, Any]],
        mode: Mode,
        **kwargs,
    ) -> Iterator[T]:
        """Stream structured output from messages."""
        ...


class ProviderBase(Generic[T]):
    """Base implementation for providers with common functionality."""

    # Class variables that should be overridden by subclasses
    _supported_modes: ClassVar[list[Mode]] = []

    @property
    def supported_modes(self) -> list[Mode]:
        """Get modes supported by this provider."""
        return self._supported_modes

    @classmethod
    def capabilities(cls) -> dict[str, bool]:
        """
        Get provider capabilities.

        Returns a dictionary with keys:
            - streaming: Whether the provider supports streaming
            - function_calling: Whether the provider supports function calling
            - async: Whether the provider supports async operations
            - multimodal: Whether the provider supports multimodal inputs
        """
        return {
            "streaming": hasattr(cls, "create_stream"),
            "function_calling": hasattr(cls, "create_with_functions"),
            "async": hasattr(cls, "acreate"),
            "multimodal": hasattr(cls, "supports_multimodal")
            and cls.supports_multimodal(),
        }

    @classmethod
    def supports_multimodal(cls) -> bool:
        """Whether this provider supports multimodal inputs."""
        return False

    @classmethod
    @abstractmethod
    def create_client(
        cls, client: Any, provider_id: Optional[str] = None, **kwargs
    ) -> Union[Instructor, AsyncInstructor]:
        """
        Create an Instructor client for this provider.

        Args:
            client: The provider's native client instance
            provider_id: Optional provider identifier (e.g., URL or name)
            **kwargs: Additional provider-specific arguments

        Returns:
            An instance of Instructor or AsyncInstructor
        """
        raise NotImplementedError()

    def validate_mode(self, mode: Mode) -> None:
        """
        Validate that the mode is supported by this provider.

        Args:
            mode: The mode to validate

        Raises:
            ValueError: If the mode is not supported
        """
        if mode not in self.supported_modes:
            supported_str = ", ".join(str(m) for m in self.supported_modes)
            raise ValueError(
                f"Mode {mode} not supported by this provider. "
                f"Supported modes: {supported_str}"
            )
