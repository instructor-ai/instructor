"""Provider-agnostic schema generation helpers owned by v2."""

from __future__ import annotations

import functools
import warnings
from typing import Any, cast

from docstring_parser import parse
from pydantic import BaseModel

from instructor.v2.providers.gemini.utils import map_to_gemini_function_schema

__all__ = [
    "generate_openai_schema",
    "generate_anthropic_schema",
    "generate_gemini_schema",
]


@functools.lru_cache(maxsize=256)
def generate_openai_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate an OpenAI function schema from a Pydantic model."""
    schema = model.model_json_schema()
    docstring = parse(model.__doc__ or "")
    parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}

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
        schema["description"] = (
            docstring.short_description
            or f"Correctly extracted `{model.__name__}` with all "
            "the required parameters with correct types"
        )

    return {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": parameters,
    }


@functools.lru_cache(maxsize=256)
def generate_anthropic_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate an Anthropic tool schema from a Pydantic model."""
    openai_schema = generate_openai_schema(model)
    return {
        "name": openai_schema["name"],
        "description": openai_schema["description"],
        "input_schema": model.model_json_schema(),
    }


@functools.lru_cache(maxsize=256)
def generate_gemini_schema(model: type[BaseModel]) -> Any:
    """Generate a legacy Gemini function schema from a Pydantic model."""
    warnings.warn(
        "generate_gemini_schema is deprecated. The google-generativeai library is being replaced by google-genai.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        import importlib

        genai_types = cast(Any, importlib.import_module("google.generativeai.types"))
        openai_schema = generate_openai_schema(model)
        return genai_types.FunctionDeclaration(
            name=openai_schema["name"],
            description=openai_schema["description"],
            parameters=map_to_gemini_function_schema(openai_schema["parameters"]),
        )
    except ImportError as e:
        raise ImportError(
            "google-generativeai is deprecated. Please install google-genai instead: pip install google-genai"
        ) from e
