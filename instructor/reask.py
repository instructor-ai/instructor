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
from instructor.utils.bedrock import reask_bedrock_json, reask_bedrock_tools
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
from instructor.utils.xai import reask_xai_json, reask_xai_tools
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

    REASK_HANDLERS = {
        Mode.ANTHROPIC_TOOLS: reask_anthropic_tools,
        Mode.ANTHROPIC_JSON: reask_anthropic_json,
        Mode.COHERE_TOOLS: reask_cohere_tools,
        Mode.GEMINI_TOOLS: reask_gemini_tools,
        Mode.GEMINI_JSON: reask_gemini_json,
        Mode.VERTEXAI_TOOLS: reask_vertexai_tools,
        Mode.VERTEXAI_JSON: reask_vertexai_json,
        Mode.TOOLS: reask_tools,
        Mode.CEREBRAS_TOOLS: reask_cerebras_tools,
        Mode.RESPONSES_TOOLS: reask_responses_tools,
        Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: reask_responses_tools,
        Mode.XAI_JSON: reask_xai_json,
        Mode.XAI_TOOLS: reask_xai_tools,
        Mode.WRITER_TOOLS: reask_writer_tools,
        Mode.WRITER_JSON: reask_writer_json,
        Mode.BEDROCK_TOOLS: reask_bedrock_tools,
        Mode.BEDROCK_JSON: reask_bedrock_json,
        Mode.PERPLEXITY_JSON: reask_perplexity_json,
        Mode.GENAI_TOOLS: reask_genai_tools,
        Mode.GENAI_STRUCTURED_OUTPUTS: reask_genai_structured_outputs,
        Mode.MISTRAL_STRUCTURED_OUTPUTS: reask_mistral_structured_outputs,
        Mode.MISTRAL_TOOLS: reask_mistral_tools,
        Mode.MD_JSON: reask_md_json,
        Mode.FIREWORKS_TOOLS: reask_fireworks_tools,
        Mode.FIREWORKS_JSON: reask_fireworks_json,
        Mode.JSON: reask_default,
        Mode.JSON_O1: reask_default,
        Mode.JSON_SCHEMA: reask_default,
        Mode.JSON_MODE: reask_default,
        Mode.PARALLEL_TOOL_CALL: reask_default,
        Mode.TOOL_CALL: reask_default,
    }

    if mode in REASK_HANDLERS:
        return REASK_HANDLERS[mode](kwargs_copy, response, exception)
    else:
        return reask_default(kwargs_copy, response, exception)
