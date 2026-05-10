"""Small generic helpers owned by the v2 runtime."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Generic, TypeVar

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
