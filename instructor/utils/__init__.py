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

# Re-export from providers
from .providers import Provider, get_provider

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
    # Gemini utils
    "transform_to_gemini_prompt",
    "verify_no_unions",
    "map_to_gemini_function_schema",
    "update_genai_kwargs",
    "update_gemini_kwargs",
    "extract_genai_system_message",
    "convert_to_genai_messages",
    # Anthropic utils
    "SystemMessage",
    "combine_system_messages",
    "extract_system_messages",
]


# Lazy imports for backward compatibility to avoid circular imports
def __getattr__(name):
    # Gemini utils
    if name in [
        "transform_to_gemini_prompt",
        "verify_no_unions",
        "map_to_gemini_function_schema",
        "update_genai_kwargs",
        "update_gemini_kwargs",
        "extract_genai_system_message",
        "convert_to_genai_messages",
    ]:
        from ..providers.gemini import utils as gemini_utils

        return getattr(gemini_utils, name)

    # Anthropic utils
    if name in [
        "SystemMessage",
        "combine_system_messages",
        "extract_system_messages",
    ]:
        from ..providers.anthropic import utils as anthropic_utils

        return getattr(anthropic_utils, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
