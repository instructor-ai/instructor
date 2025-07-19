"""Utility modules for instructor library.

This package contains utility functions organized by provider and functionality.
"""

# Re-export everything from core
from .core import (
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
    update_total_usage,
    dump_message,
    is_async,
    merge_consecutive_messages,
    classproperty,
    get_message_content,
    disable_pydantic_error_url,
    is_typed_dict,
    is_simple_type,
    prepare_response_model,
)

# Note: Provider and get_provider are in providers.py
# Import them directly from there when needed to avoid circular imports

# Note: anthropic utils are now in providers/anthropic/utils.py
# Import them directly from there when needed

# Note: google utils are in google.py
# Import them directly from there when needed to avoid circular imports

__all__ = [
    # Core functions
    "extract_json_from_codeblock",
    "extract_json_from_stream",
    "extract_json_from_stream_async",
    "update_total_usage",
    "dump_message",
    "is_async",
    "merge_consecutive_messages",
    "classproperty",
    "get_message_content",
    "disable_pydantic_error_url",
    "is_typed_dict",
    "is_simple_type",
    "prepare_response_model",
]
