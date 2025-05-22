"""Base provider abstraction for instructor providers."""

from abc import ABC, abstractmethod
from typing import Any, Union, TypeVar
from collections.abc import Awaitable
from pydantic import BaseModel

from ..mode import Mode
from ..client import Instructor, AsyncInstructor

T_Model = TypeVar("T_Model", bound=BaseModel)


class BaseProvider(ABC):
    """Abstract base class for all instructor providers."""

    @abstractmethod
    def get_supported_modes(self) -> set[Mode]:
        """Return the set of modes supported by this provider."""
        pass

    @abstractmethod
    def validate_response(self, response: Any, mode: Mode) -> None:
        """Validate provider-specific response format."""
        pass

    @abstractmethod
    def process_response(
        self, response: Any, response_model: type[T_Model], mode: Mode, **kwargs: Any
    ) -> T_Model:
        """Process provider response and return structured output."""
        pass

    @abstractmethod
    def process_response_async(
        self, response: Any, response_model: type[T_Model], mode: Mode, **kwargs: Any
    ) -> Awaitable[T_Model]:
        """Process provider response asynchronously and return structured output."""
        pass

    @abstractmethod
    def create_instructor(
        self, client: Any, mode: Mode, **kwargs: Any
    ) -> Union[Instructor, AsyncInstructor]:
        """Create an instructor instance for this provider."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    def supports_mode(self, mode: Mode) -> bool:
        """Check if this provider supports the given mode."""
        return mode in self.get_supported_modes()
