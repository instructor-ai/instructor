"""
Standalone schema generation utilities for different LLM providers.

This module provides provider-agnostic functions to generate schemas from Pydantic models
without requiring inheritance from OpenAISchema or use of decorators.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Optional

from docstring_parser import parse
from pydantic import BaseModel

from ..providers.gemini.utils import map_to_gemini_function_schema
from ..mode import Mode

__all__ = [
    "generate_openai_schema",
    "generate_anthropic_schema",
    "generate_gemini_schema",
]


@functools.lru_cache(maxsize=256)
def generate_openai_schema(model: type[BaseModel], mode: Optional[Mode] = None) -> dict[str, Any]:
    """
    Generate OpenAI function schema from a Pydantic model.

    Args:
        model: A Pydantic BaseModel subclass
        mode: Optional Mode parameter for mode-specific schema variations.
              Currently unused but reserved for future mode-specific schema formats.

    Returns:
        A dictionary in the format of OpenAI's function schema

    Note:
        The model's docstring will be used for the function description.
        Parameter descriptions from the docstring will enrich field descriptions.
        The mode parameter is included in the cache key for potential future mode-specific variations.
    """
    # Mode is included in cache key via functools.lru_cache based on function arguments
    # No mode-specific logic yet, but infrastructure is in place for future enhancements
    _ = mode  # Reserved for future use
    
    schema = model.model_json_schema()
    docstring = parse(model.__doc__ or "")
    parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}

    # Enrich parameter descriptions from docstring
    for param in docstring.params:
        if (name := param.arg_name) in parameters["properties"] and (
            description := param.description
        ):
            if "description" not in parameters["properties"][name]:
                parameters["properties"][name]["description"] = description

    parameters["required"] = sorted(
        k for k, v in parameters["properties"].items() if "default" not in v
    )

    if "description" not in schema:
        if docstring.short_description:
            schema["description"] = docstring.short_description
        else:
            schema["description"] = (
                f"Correctly extracted `{model.__name__}` with all "
                f"the required parameters with correct types"
            )

    return {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": parameters,
    }


@functools.lru_cache(maxsize=256)
def generate_anthropic_schema(model: type[BaseModel], mode: Optional[Mode] = None) -> dict[str, Any]:
    """
    Generate Anthropic tool schema from a Pydantic model.

    Args:
        model: A Pydantic BaseModel subclass
        mode: Optional Mode parameter for mode-specific schema variations.
              Currently unused but reserved for future mode-specific schema formats.

    Returns:
        A dictionary in the format of Anthropic's tool schema
    """
    # Mode is included in cache key via functools.lru_cache based on function arguments
    # No mode-specific logic yet, but infrastructure is in place for future enhancements
    _ = mode  # Reserved for future use
    
    # Generate the Anthropic schema based on the OpenAI schema to avoid redundant schema generation
    openai_schema = generate_openai_schema(model, mode)
    return {
        "name": openai_schema["name"],
        "description": openai_schema["description"],
        "input_schema": model.model_json_schema(),
    }


@functools.lru_cache(maxsize=256)
def generate_gemini_schema(model: type[BaseModel], mode: Optional[Mode] = None) -> Any:
    """
    Generate Gemini function schema from a Pydantic model.

    Args:
        model: A Pydantic BaseModel subclass
        mode: Optional Mode parameter for mode-specific schema variations.
              Currently unused but reserved for future mode-specific schema formats.

    Returns:
        A Gemini FunctionDeclaration object

    Note:
        This function is deprecated. The google-generativeai library is being replaced by google-genai.
    """
    # Mode is included in cache key via functools.lru_cache based on function arguments
    # No mode-specific logic yet, but infrastructure is in place for future enhancements
    _ = mode  # Reserved for future use
    
    # This is kept for backward compatibility but deprecated
    warnings.warn(
        "generate_gemini_schema is deprecated. The google-generativeai library is being replaced by google-genai.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        import google.generativeai.types as genai_types

        # Use OpenAI schema
        openai_schema = generate_openai_schema(model, mode)

        # Transform to Gemini format
        function = genai_types.FunctionDeclaration(
            name=openai_schema["name"],
            description=openai_schema["description"],
            parameters=map_to_gemini_function_schema(openai_schema["parameters"]),
        )

        return function
    except ImportError as e:
        raise ImportError(
            "google-generativeai is deprecated. Please install google-genai instead: pip install google-genai"
        ) from e
