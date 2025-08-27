"""Core components of the instructor package."""

import warnings
from .client import Instructor, AsyncInstructor, Response, from_openai, from_litellm
from .hooks import Hooks, HookName
from .patch import patch, apatch
from .retry import retry_sync, retry_async

__all__ = [
    "Instructor",
    "AsyncInstructor",
    "Response",
    "Hooks",
    "HookName",
    "patch",
    "apatch",
    "from_openai",
    "from_litellm",
    "retry_sync",
    "retry_async",
]

_DEPRECATED_EXCEPTIONS = {
    "InstructorRetryException",
    "InstructorError",
    "ConfigurationError",
    "IncompleteOutputException",
    "ValidationError",
    "ProviderError",
    "ModeError",
    "ClientError",
    "AsyncValidationError",
}

def __getattr__(name: str):
    if name in _DEPRECATED_EXCEPTIONS:
        warnings.warn(
            f"Importing {name} from 'instructor.core' is deprecated. "
            f"Please use 'from instructor.exceptions import {name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from .exceptions import (
            InstructorRetryException,
            InstructorError,
            ConfigurationError,
            IncompleteOutputException,
            ValidationError,
            ProviderError,
            ModeError,
            ClientError,
            AsyncValidationError,
        )
        exception_map = {
            "InstructorRetryException": InstructorRetryException,
            "InstructorError": InstructorError,
            "ConfigurationError": ConfigurationError,
            "IncompleteOutputException": IncompleteOutputException,
            "ValidationError": ValidationError,
            "ProviderError": ProviderError,
            "ModeError": ModeError,
            "ClientError": ClientError,
            "AsyncValidationError": AsyncValidationError,
        }
        return exception_map[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
