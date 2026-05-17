"""Small generic helpers owned by the v2 runtime."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic import ValidationError

R_co = TypeVar("R_co", covariant=True)


def is_async(func: Callable[..., Any]) -> bool:
    """Return whether a callable is async, following wrapped callables."""
    is_coroutine = inspect.iscoroutinefunction(func)
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__  # type: ignore[attr-defined]
        is_coroutine = is_coroutine or inspect.iscoroutinefunction(func)
    return is_coroutine


class classproperty(Generic[R_co]):
    """Descriptor for class-level properties."""

    def __init__(self, method: Callable[[Any], R_co]) -> None:
        self.cproperty = method

    def __get__(self, instance: object, cls: type[Any]) -> R_co:
        return self.cproperty(cls)


def disable_pydantic_error_url() -> None:
    """Disable URLs in Pydantic ValidationError messages."""
    if not hasattr(ValidationError, "_original_str"):
        ValidationError._original_str = ValidationError.__str__  # type: ignore[attr-defined]

    def __str__(self: ValidationError) -> str:
        output = ValidationError._original_str(self)  # type: ignore[attr-defined]
        return "\n".join(
            line
            for line in output.split("\n")
            if "https://errors.pydantic.dev" not in line
        )

    ValidationError.__str__ = __str__  # type: ignore[method-assign]
