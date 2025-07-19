"""Backward compatibility module for instructor.function_calls imports."""

# Re-export everything from the new location
from .processing.function_calls import *  # noqa: F403, F401

# Explicitly re-export internal functions for backward compatibility
from .processing.function_calls import (
    _extract_text_content,
    _validate_model_from_json,
)

__all__ = ["_extract_text_content", "_validate_model_from_json"]

# This allows `from instructor.function_calls import OpenAISchema` to work
