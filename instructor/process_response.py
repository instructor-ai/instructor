"""Response processing module for instructor.

This module maintains backward compatibility while delegating to the new
refactored response_processing package.
"""

# Import everything from the compatibility module to maintain backward compatibility
from instructor.process_response_compat import *  # noqa: F403, F401

# Also import specific items that might be used with explicit imports
from instructor.process_response_compat import (
    handle_anthropic_json,
    handle_anthropic_tools,
    handle_bedrock_json,
    handle_bedrock_tools,
    handle_cohere_json_schema,
    handle_cohere_tools,
    handle_functions,
    handle_gemini_json,
    handle_gemini_tools,
    handle_json_modes,
    handle_mistral_structured_outputs,
    handle_mistral_tools,
    handle_parallel_model,
    handle_parallel_tools,
    handle_response_model,
    handle_tools,
    handle_tools_strict,
    handle_vertexai_json,
    handle_vertexai_parallel_tools,
    handle_vertexai_tools,
    is_typed_dict,
    prepare_response_model,
    process_response,
    process_response_async,
)

__all__ = [
    "process_response",
    "process_response_async",
    "is_typed_dict",
    "prepare_response_model",
    "handle_response_model",
    "handle_parallel_model",
    "handle_functions",
    "handle_tools_strict",
    "handle_tools",
    "handle_parallel_tools",
    "handle_json_modes",
    "handle_anthropic_tools",
    "handle_anthropic_json",
    "handle_gemini_json",
    "handle_gemini_tools",
    "handle_bedrock_json",
    "handle_bedrock_tools",
    "handle_mistral_tools",
    "handle_mistral_structured_outputs",
    "handle_cohere_json_schema",
    "handle_cohere_tools",
    "handle_vertexai_json",
    "handle_vertexai_tools",
    "handle_vertexai_parallel_tools",
]
