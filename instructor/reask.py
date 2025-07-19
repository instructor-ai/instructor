from __future__ import annotations

import logging
from typing import Any, TypeVar

from instructor.mode import Mode
from pydantic import BaseModel
from typing_extensions import ParamSpec

# Import reask functions organized by provider (matching process_response.py structure)

# OpenAI reask functions
from instructor.utils.openai import (
    reask_default,
    reask_md_json,
    reask_responses_tools,
    reask_tools,
)

# Anthropic reask functions
from instructor.utils.anthropic import (
    reask_anthropic_json,
    reask_anthropic_tools,
)

# Google/Gemini reask functions
from instructor.utils.google import (
    reask_gemini_json,
    reask_gemini_tools,
    reask_genai_structured_outputs,
    reask_genai_tools,
    reask_vertexai_json,
    reask_vertexai_tools,
)

# Mistral reask functions
from instructor.utils.mistral import (
    reask_mistral_structured_outputs,
    reask_mistral_tools,
)

# Cohere reask functions
from instructor.utils.cohere import reask_cohere_tools

# Cerebras reask functions
from instructor.utils.cerebras import reask_cerebras_tools

# Fireworks reask functions
from instructor.utils.fireworks import reask_fireworks_json, reask_fireworks_tools

# Writer reask functions
from instructor.utils.writer import reask_writer_json, reask_writer_tools

# Bedrock reask functions
from instructor.utils.bedrock import reask_bedrock_json, reask_bedrock_tools

# Perplexity reask functions
from instructor.utils.perplexity import reask_perplexity_json

# XAI reask functions
from instructor.utils.xai import reask_xai_json, reask_xai_tools

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


def handle_reask_kwargs(
    kwargs: dict[str, Any],
    mode: Mode,
    response: Any,  # Replace with actual response type based on the mode
    exception: Exception,
):
    # Create a shallow copy of kwargs to avoid modifying the original
    kwargs_copy = kwargs.copy()

    # Organized by provider (matching process_response.py structure)
    REASK_HANDLERS = {
        # OpenAI modes
        Mode.FUNCTIONS: reask_default,
        Mode.TOOLS_STRICT: reask_tools,
        Mode.TOOLS: reask_tools,
        Mode.JSON_O1: reask_default,
        Mode.JSON: reask_md_json,
        Mode.MD_JSON: reask_md_json,
        Mode.JSON_SCHEMA: reask_md_json,
        Mode.PARALLEL_TOOLS: reask_tools,
        Mode.RESPONSES_TOOLS: reask_responses_tools,
        Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: reask_responses_tools,
        # Mistral modes
        Mode.MISTRAL_TOOLS: reask_mistral_tools,
        Mode.MISTRAL_STRUCTURED_OUTPUTS: reask_mistral_structured_outputs,
        # Anthropic modes
        Mode.ANTHROPIC_TOOLS: reask_anthropic_tools,
        Mode.ANTHROPIC_REASONING_TOOLS: reask_anthropic_tools,
        Mode.ANTHROPIC_JSON: reask_anthropic_json,
        Mode.ANTHROPIC_PARALLEL_TOOLS: reask_anthropic_tools,
        # Cohere modes
        Mode.COHERE_TOOLS: reask_cohere_tools,
        Mode.COHERE_JSON_SCHEMA: reask_cohere_tools,
        # Gemini/Google modes
        Mode.GEMINI_TOOLS: reask_gemini_tools,
        Mode.GEMINI_JSON: reask_gemini_json,
        Mode.GENAI_TOOLS: reask_genai_tools,
        Mode.GENAI_STRUCTURED_OUTPUTS: reask_genai_structured_outputs,
        # VertexAI modes
        Mode.VERTEXAI_TOOLS: reask_vertexai_tools,
        Mode.VERTEXAI_JSON: reask_vertexai_json,
        Mode.VERTEXAI_PARALLEL_TOOLS: reask_vertexai_tools,
        # Cerebras modes
        Mode.CEREBRAS_TOOLS: reask_cerebras_tools,
        Mode.CEREBRAS_JSON: reask_default,
        # Fireworks modes
        Mode.FIREWORKS_TOOLS: reask_fireworks_tools,
        Mode.FIREWORKS_JSON: reask_fireworks_json,
        # Writer modes
        Mode.WRITER_TOOLS: reask_writer_tools,
        Mode.WRITER_JSON: reask_writer_json,
        # Bedrock modes
        Mode.BEDROCK_TOOLS: reask_bedrock_tools,
        Mode.BEDROCK_JSON: reask_bedrock_json,
        # Perplexity modes
        Mode.PERPLEXITY_JSON: reask_perplexity_json,
        # OpenRouter modes
        Mode.OPENROUTER_STRUCTURED_OUTPUTS: reask_default,
        # XAI modes
        Mode.XAI_JSON: reask_xai_json,
        Mode.XAI_TOOLS: reask_xai_tools,
    }

    if mode in REASK_HANDLERS:
        return REASK_HANDLERS[mode](kwargs_copy, response, exception)
    else:
        return reask_default(kwargs_copy, response, exception)
