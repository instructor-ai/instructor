"""Utility modules for instructor library.

This package contains utility functions organized by provider and functionality.
"""

# Re-export everything from core
from instructor.utils.core import (
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

# Re-export from providers
from instructor.utils.providers import Provider, get_provider

# Re-export from anthropic
from instructor.utils.anthropic import (
    SystemMessage,
    combine_system_messages,
    extract_system_messages,
)

# Re-export from google
from instructor.utils.google import (
    transform_to_gemini_prompt,
    verify_no_unions,
    map_to_gemini_function_schema,
    update_genai_kwargs,
    update_gemini_kwargs,
    extract_genai_system_message,
    convert_to_genai_messages,
)

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
    # Provider functions
    "Provider",
    "get_provider",
    # Anthropic functions
    "SystemMessage",
    "combine_system_messages",
    "extract_system_messages",
    # Google functions
    "transform_to_gemini_prompt",
    "verify_no_unions",
    "map_to_gemini_function_schema",
    "update_genai_kwargs",
    "update_gemini_kwargs",
    "extract_genai_system_message",
    "convert_to_genai_messages",
]
