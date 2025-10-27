"""
ListResponse - A list-like wrapper that can hold a raw response.

This module provides a list-like container that supports attaching
a raw LLM response, enabling proper usage with create_with_completion().
"""

from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


class ListResponse(list[T], Generic[T]):
    """
    A list-like container that can hold a raw API response.

    This wrapper allows us to return a list from create_with_completion()
    while still preserving the raw LLM response for access to metadata
    like token usage, model info, etc.

    The class behaves like a normal Python list in all respects, but
    additionally supports storing and accessing a _raw_response attribute.

    Example:
        ```python
        from instructor.dsl import ListResponse
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        # Works with create_with_completion()
        users, completion = client.chat.completions.create_with_completion(
            response_model=list[User],
            messages=[...],
        )
        # users is a ListResponse[User]
        # completion contains token usage, model, etc.
        print(len(users))  # Works like a normal list
        print(users[0])    # Access by index
        print(users._raw_response)  # Access the raw response
        ```

    Note:
        This class is automatically used by instructor when returning
        Iterable or List responses. You don't need to instantiate it directly.
    """

    _raw_response: Any = None

    def __init__(self, iterable=None, _raw_response: Any = None):
        """
        Initialize ListResponse.

        Args:
            iterable: Optional iterable to initialize the list with
            _raw_response: Optional raw API response to attach
        """
        if iterable is not None:
            super().__init__(iterable)
        else:
            super().__init__()
        self._raw_response = _raw_response

    def __repr__(self) -> str:
        """Return a string representation."""
        items_repr = list.__repr__(self)
        if self._raw_response is not None:
            return f"ListResponse({items_repr}, _raw_response=...)"
        return f"ListResponse({items_repr})"

    def __str__(self) -> str:
        """Return a string representation."""
        return list.__str__(self)

    @classmethod
    def from_list(cls, items: list[T], raw_response: Any = None) -> "ListResponse[T]":
        """
        Create a ListResponse from a list and optional raw response.

        Args:
            items: List of items
            raw_response: Optional raw API response

        Returns:
            ListResponse instance with items and raw response attached
        """
        response_list = cls(items, _raw_response=raw_response)
        return response_list

    def get_raw_response(self) -> Optional[Any]:
        """
        Get the attached raw response.

        Returns:
            The raw API response, or None if not attached
        """
        return self._raw_response


# Type alias for convenience
ResponseList = ListResponse
