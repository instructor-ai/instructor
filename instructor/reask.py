from __future__ import annotations

import logging
from typing import Any, TypeVar

from instructor.mode import Mode
from pydantic import BaseModel
from typing_extensions import ParamSpec

# Import reask functions from provider-specific modules
from instructor.utils.anthropic import (
    reask_anthropic_json,
    reask_anthropic_tools,
)
from instructor.utils.bedrock import reask_bedrock_json
from instructor.utils.cerebras import reask_cerebras_tools
from instructor.utils.cohere import reask_cohere_tools
from instructor.utils.fireworks import reask_fireworks_json, reask_fireworks_tools
from instructor.utils.google import (
    reask_gemini_json,
    reask_gemini_tools,
    reask_genai_structured_outputs,
    reask_genai_tools,
    reask_vertexai_json,
    reask_vertexai_tools,
)
from instructor.utils.mistral import (
    reask_mistral_structured_outputs,
    reask_mistral_tools,
)
from instructor.utils.openai import (
    reask_default,
    reask_md_json,
    reask_responses_tools,
    reask_tools,
)
from instructor.utils.perplexity import reask_perplexity_json
from instructor.utils.writer import reask_writer_json, reask_writer_tools

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

    # Use a more efficient mapping approach with mode groupings to reduce lookup time
    # Group similar modes that use the same reask function
    if mode in {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_REASONING_TOOLS}:
        return reask_anthropic_tools(kwargs_copy, response, exception)
    elif mode == Mode.ANTHROPIC_JSON:
        return reask_anthropic_json(kwargs_copy, response, exception)
    elif mode in {Mode.COHERE_TOOLS, Mode.COHERE_JSON_SCHEMA}:
        return reask_cohere_tools(kwargs_copy, response, exception)
    elif mode == Mode.GEMINI_TOOLS:
        return reask_gemini_tools(kwargs_copy, response, exception)
    elif mode == Mode.GEMINI_JSON:
        return reask_gemini_json(kwargs_copy, response, exception)
    elif mode == Mode.VERTEXAI_TOOLS:
        return reask_vertexai_tools(kwargs_copy, response, exception)
    elif mode == Mode.VERTEXAI_JSON:
        return reask_vertexai_json(kwargs_copy, response, exception)
    elif mode in {Mode.TOOLS, Mode.TOOLS_STRICT}:
        return reask_tools(kwargs_copy, response, exception)
    elif mode == Mode.CEREBRAS_TOOLS:
        return reask_cerebras_tools(kwargs_copy, response, exception)
    elif mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}:
        return reask_responses_tools(kwargs_copy, response, exception)
    elif mode == Mode.MD_JSON:
        return reask_md_json(kwargs_copy, response, exception)
    elif mode == Mode.FIREWORKS_TOOLS:
        return reask_fireworks_tools(kwargs_copy, response, exception)
    elif mode == Mode.FIREWORKS_JSON:
        return reask_fireworks_json(kwargs_copy, response, exception)
    elif mode == Mode.WRITER_TOOLS:
        return reask_writer_tools(kwargs_copy, response, exception)
    elif mode == Mode.WRITER_JSON:
        return reask_writer_json(kwargs_copy, response, exception)
    elif mode == Mode.BEDROCK_JSON:
        return reask_bedrock_json(kwargs_copy, response, exception)
    elif mode == Mode.PERPLEXITY_JSON:
        return reask_perplexity_json(kwargs_copy, response, exception)
    elif mode == Mode.GENAI_TOOLS:
        return reask_genai_tools(kwargs_copy, response, exception)
    elif mode == Mode.GENAI_STRUCTURED_OUTPUTS:
        return reask_genai_structured_outputs(kwargs_copy, response, exception)
    elif mode == Mode.MISTRAL_STRUCTURED_OUTPUTS:
        return reask_mistral_structured_outputs(kwargs_copy, response, exception)
    elif mode == Mode.MISTRAL_TOOLS:
        return reask_mistral_tools(kwargs_copy, response, exception)
    else:
        return reask_default(kwargs_copy, response, exception)
