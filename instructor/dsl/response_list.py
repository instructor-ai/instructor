"""List-like response wrapper.

When a response model is a list (for example `list[User]`), we still want to
attach the provider's raw response so `create_with_completion()` can return it.

This module provides `ListResponse`, a normal Python `list` with an extra
`_raw_response` attribute.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ListResponse(list[T], Generic[T]):
    """A list that also stores the raw provider response."""

    _raw_response: Any = None

    def __init__(self, iterable: Any = None, _raw_response: Any = None):
        if iterable is None:
            super().__init__()
        else:
            super().__init__(iterable)
        self._raw_response = _raw_response

    @classmethod
    def from_list(cls, items: list[T], raw_response: Any = None) -> ListResponse[T]:
        """Create a `ListResponse` from items and an optional raw response."""

        return cls(items, _raw_response=raw_response)


# Backwards-friendly alias
ResponseList = ListResponse
