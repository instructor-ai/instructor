"""OpenAI-specific schema helpers."""

from __future__ import annotations

import functools
from typing import Any

from docstring_parser import parse
from pydantic import BaseModel


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
