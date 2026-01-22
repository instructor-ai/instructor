"""List-like response wrapper.

When a response model returns a list (for example `list[User]`), we still want to
attach the provider's raw response so `create_with_completion()` can return it.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ListResponse(list[T], Generic[T]):
    """A list that preserves the underlying provider response.

    This is used when a call returns a list of objects (e.g. `list[User]`), so
    `create_with_completion()` can still return `(result, raw_response)` without
    crashing on a plain `list`.
    """

    _raw_response: Any | None

    def __init__(self, iterable=(), _raw_response: Any | None = None):  # type: ignore[no-untyped-def]
        super().__init__(iterable)
        self._raw_response = _raw_response

    @classmethod
    def from_list(cls, items: list[T], *, raw_response: Any | None) -> ListResponse[T]:
        return cls(items, _raw_response=raw_response)

    def get_raw_response(self) -> Any | None:
        return self._raw_response

    def __getitem__(self, key):  # type: ignore[no-untyped-def]
        value = super().__getitem__(key)
        if isinstance(key, slice):
            return type(self)(value, _raw_response=self._raw_response)
        return value


# Backwards-friendly alias
ResponseList = ListResponse
