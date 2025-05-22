"""Backward compatibility module for process_response.

This module provides backward compatibility by re-exporting all functions
from the refactored response_processing module to maintain the original API.
"""

# Import all public functions from the refactored modules
from instructor.response_processing import (
    handle_response_model,
    is_typed_dict,
    prepare_response_model,
    process_response,
    process_response_async,
)

# Import handler functions that were previously in process_response.py
from instructor.dsl.parallel import handle_parallel_model
from instructor.response_processing.providers.openai import (
    OpenAIFunctionsHandler,
    OpenAIJSONHandler,
    OpenAIParallelToolsHandler,
    OpenAIToolsHandler,
    OpenAIToolsStrictHandler,
)
from instructor.response_processing.providers.anthropic import (
    AnthropicJSONHandler,
    AnthropicToolsHandler,
)
from instructor.response_processing.providers.gemini import (
    GeminiJSONHandler,
    GeminiToolsHandler,
)
from instructor.response_processing.providers.bedrock import (
    BedrockJSONHandler,
    BedrockToolsHandler,
)
from instructor.response_processing.providers.mistral import (
    MistralStructuredOutputsHandler,
    MistralToolsHandler,
)
from instructor.response_processing.providers.cohere import (
    CohereJSONSchemaHandler,
    CohereToolsHandler,
)
from instructor.response_processing.providers.vertexai import (
    VertexAIJSONHandler,
    VertexAIParallelToolsHandler,
    VertexAIToolsHandler,
)

# Create handler instances to mimic the old function-based approach
_tools_handler = OpenAIToolsHandler()
_tools_strict_handler = OpenAIToolsStrictHandler()
_functions_handler = OpenAIFunctionsHandler()
_json_handler = OpenAIJSONHandler
_parallel_tools_handler = OpenAIParallelToolsHandler()
_anthropic_tools_handler = AnthropicToolsHandler()
_anthropic_json_handler = AnthropicJSONHandler()
_gemini_json_handler = GeminiJSONHandler()
_gemini_tools_handler = GeminiToolsHandler()
_bedrock_json_handler = BedrockJSONHandler()
_bedrock_tools_handler = BedrockToolsHandler()
_mistral_tools_handler = MistralToolsHandler()
_mistral_structured_handler = MistralStructuredOutputsHandler()
_cohere_json_handler = CohereJSONSchemaHandler()
_cohere_tools_handler = CohereToolsHandler()
_vertexai_json_handler = VertexAIJSONHandler()
_vertexai_tools_handler = VertexAIToolsHandler()
_vertexai_parallel_handler = VertexAIParallelToolsHandler()


# Recreate the old function signatures for backward compatibility
def handle_functions(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_functions."""
    return _functions_handler.handle(response_model, new_kwargs)


def handle_tools_strict(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_tools_strict."""
    return _tools_strict_handler.handle(response_model, new_kwargs)


def handle_tools(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_tools."""
    return _tools_handler.handle(response_model, new_kwargs)


def handle_parallel_tools(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_parallel_tools."""
    return _parallel_tools_handler.handle(response_model, new_kwargs)


def handle_json_modes(response_model, new_kwargs, mode):
    """Backward compatibility wrapper for handle_json_modes."""
    handler = _json_handler(mode)
    return handler.handle(response_model, new_kwargs)


def handle_anthropic_tools(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_anthropic_tools."""
    return _anthropic_tools_handler.handle(response_model, new_kwargs)


def handle_anthropic_json(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_anthropic_json."""
    return _anthropic_json_handler.handle(response_model, new_kwargs)


def handle_gemini_json(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_gemini_json."""
    return _gemini_json_handler.handle(response_model, new_kwargs)


def handle_gemini_tools(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_gemini_tools."""
    return _gemini_tools_handler.handle(response_model, new_kwargs)


def handle_bedrock_json(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_bedrock_json."""
    return _bedrock_json_handler.handle(response_model, new_kwargs)


def handle_bedrock_tools(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_bedrock_tools."""
    return _bedrock_tools_handler.handle(response_model, new_kwargs)


def handle_mistral_tools(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_mistral_tools."""
    return _mistral_tools_handler.handle(response_model, new_kwargs)


def handle_mistral_structured_outputs(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_mistral_structured_outputs."""
    return _mistral_structured_handler.handle(response_model, new_kwargs)


def handle_cohere_json_schema(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_cohere_json_schema."""
    return _cohere_json_handler.handle(response_model, new_kwargs)


def handle_cohere_tools(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_cohere_tools."""
    return _cohere_tools_handler.handle(response_model, new_kwargs)


def handle_vertexai_json(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_vertexai_json."""
    return _vertexai_json_handler.handle(response_model, new_kwargs)


def handle_vertexai_tools(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_vertexai_tools."""
    return _vertexai_tools_handler.handle(response_model, new_kwargs)


def handle_vertexai_parallel_tools(response_model, new_kwargs):
    """Backward compatibility wrapper for handle_vertexai_parallel_tools."""
    return _vertexai_parallel_handler.handle(response_model, new_kwargs)


# Export all functions to maintain backward compatibility
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
