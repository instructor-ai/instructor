from __future__ import annotations

from typing import Any


class IncompleteOutputException(Exception):
    """Exception raised when the output from LLM is incomplete due to max tokens limit reached."""

    def __init__(
        self,
        *args: list[Any],
        last_completion: Any | None = None,
        message: str = "The output is incomplete due to a max_tokens length limit.",
        **kwargs: dict[str, Any],
    ):
        self.last_completion = last_completion
        super().__init__(message, *args, **kwargs)
