"""Response processing module for instructor."""

from .core import (
    is_typed_dict,
    prepare_response_model,
    process_response,
    process_response_async,
)
from .handler import handle_response_model
from .registry import HandlerRegistry

__all__ = [
    "process_response",
    "process_response_async",
    "prepare_response_model",
    "is_typed_dict",
    "handle_response_model",
    "HandlerRegistry",
]
