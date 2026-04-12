"""Core components of the instructor package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Instructor",
    "AsyncInstructor",
    "Response",
    "InstructorRetryException",
    "InstructorError",
    "ConfigurationError",
    "IncompleteOutputException",
    "ValidationError",
    "ProviderError",
    "ModeError",
    "ClientError",
    "AsyncValidationError",
    "FailedAttempt",
    "ResponseParsingError",
    "MultimodalError",
    "Hooks",
    "HookName",
    "patch",
    "apatch",
    "from_openai",
    "from_litellm",
    "retry_sync",
    "retry_async",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Instructor": (".client", "Instructor"),
    "AsyncInstructor": (".client", "AsyncInstructor"),
    "Response": (".client", "Response"),
    "from_openai": (".client", "from_openai"),
    "from_litellm": (".client", "from_litellm"),
    "InstructorRetryException": (".exceptions", "InstructorRetryException"),
    "InstructorError": (".exceptions", "InstructorError"),
    "ConfigurationError": (".exceptions", "ConfigurationError"),
    "IncompleteOutputException": (".exceptions", "IncompleteOutputException"),
    "ValidationError": (".exceptions", "ValidationError"),
    "ProviderError": (".exceptions", "ProviderError"),
    "ModeError": (".exceptions", "ModeError"),
    "ClientError": (".exceptions", "ClientError"),
    "AsyncValidationError": (".exceptions", "AsyncValidationError"),
    "FailedAttempt": (".exceptions", "FailedAttempt"),
    "ResponseParsingError": (".exceptions", "ResponseParsingError"),
    "MultimodalError": (".exceptions", "MultimodalError"),
    "Hooks": (".hooks", "Hooks"),
    "HookName": (".hooks", "HookName"),
    "patch": (".patch", "patch"),
    "apatch": (".patch", "apatch"),
    "retry_sync": (".retry", "retry_sync"),
    "retry_async": (".retry", "retry_async"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_path, package=__name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
