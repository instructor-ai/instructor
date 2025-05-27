"""
Base provider interface for Instructor.

This module defines the base interface that all provider implementations should follow.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol, TypeVar, Union

import instructor
from instructor.utils import Provider


class ProviderBase(ABC):
    """Base class for all provider implementations."""
    
    @abstractmethod
    def from_client(self, client: Any, mode: instructor.Mode, **kwargs: Any) -> Union[instructor.Instructor, instructor.AsyncInstructor]:
        """
        Create an Instructor instance from a provider client.
        
        Args:
            client: An instance of the provider's client (sync or async)
            mode: The mode to use for the client
            **kwargs: Additional keyword arguments to pass to the Instructor constructor
            
        Returns:
            An Instructor instance (sync or async depending on the client type)
        """
        pass
