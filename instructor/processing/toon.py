"""TOON (Token-Oriented Object Notation) processing utilities.

This module contains shared utilities for TOON mode across all providers,
including structure generation, prompt templates, and reask message formatting.

TOON achieves 30-60% token reduction compared to JSON while maintaining
structured outputs through a compact, YAML-like format.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any
import typing


def _format_type_for_toon(annotation: Any, description: str) -> str:
    """Format a type annotation for TOON structure display."""
    from enum import Enum

    origin = getattr(annotation, "__origin__", None)

    if origin is typing.Annotated:
        args = getattr(annotation, "__args__", ())
        if args:
            return _format_type_for_toon(args[0], description)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        choices = "|".join(str(m.value) for m in annotation)
        return f"<{choices}>"

    if origin is typing.Literal:
        choices = "|".join(str(v) for v in getattr(annotation, "__args__", ()))
        return f"<{choices}>"

    if origin is typing.Union:
        args = [
            arg for arg in getattr(annotation, "__args__", ()) if arg is not type(None)
        ]
        type_names = []
        for t in args:
            if t is str:
                type_names.append("str")
            elif t is int:
                type_names.append("int")
            elif t is float:
                type_names.append("float")
            elif t is bool:
                type_names.append("bool")
            elif isinstance(t, type) and issubclass(t, Enum):
                type_names.append("|".join(str(m.value) for m in t))
            else:
                type_names.append(str(t.__name__) if hasattr(t, "__name__") else str(t))
        return f"<{' or '.join(type_names)}>"

    if annotation is str:
        return f'"<str>"'
    elif annotation is int:
        return "<int>"
    elif annotation is float:
        return "<float>"
    elif annotation is bool:
        return "<bool>"
    else:
        return f"<{description}>"


def generate_toon_structure(model: type[Any], indent: int = 0) -> str:
    """
    Generate a TOON structure template from a Pydantic model.

    Recursively expands nested Pydantic models to show full structure.
    Handles Enums, Literals, Unions, Annotated, and nested types.

    Args:
        model: A Pydantic BaseModel class
        indent: Current indentation level

    Returns:
        A string representing the TOON structure template
    """
    from pydantic import BaseModel

    prefix = "  " * indent
    lines = []

    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        description = field_info.description or f"value for {field_name}"

        origin = getattr(annotation, "__origin__", None)
        if origin is typing.Annotated:
            args = getattr(annotation, "__args__", ())
            if args:
                annotation = args[0]
                origin = getattr(annotation, "__origin__", None)

        original_annotation = annotation

        if origin is type(None) or str(origin) == "typing.Union":
            args = getattr(annotation, "__args__", ())
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                annotation = non_none_args[0]
                original_annotation = non_none_args[0]
            elif len(non_none_args) > 1:
                formatted = _format_type_for_toon(annotation, description)
                lines.append(f"{prefix}{field_name}: {formatted}")
                continue

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            lines.append(f"{prefix}{field_name}:")
            nested_structure = generate_toon_structure(annotation, indent + 1)
            lines.append(nested_structure)
        elif origin is list:
            args = getattr(annotation, "__args__", ())
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                item_model = args[0]
                has_nested_list = any(
                    getattr(f.annotation, "__origin__", None) is list
                    for f in item_model.model_fields.values()
                )
                if has_nested_list:
                    lines.append(f"{prefix}{field_name}[N]:")
                    lines.append(f"{prefix}  - <item>:")
                    nested = generate_toon_structure(item_model, indent + 2)
                    lines.append(nested)
                    lines.append(f"{prefix}  ...")
                else:
                    item_fields = list(item_model.model_fields.keys())
                    headers = ",".join(item_fields)
                    lines.append(f"{prefix}{field_name}[N,]{{{headers}}}:")
                    placeholders = ",".join(
                        f"<{item_model.model_fields[f].description or f}>"
                        for f in item_fields
                    )
                    lines.append(f"{prefix}  {placeholders}")
                    lines.append(f"{prefix}  ...")
            else:
                lines.append(f"{prefix}{field_name}[N]: <value>,<value>,...")
        elif origin is dict or annotation is dict:
            lines.append(f"{prefix}{field_name}:")
            lines.append(f"{prefix}  <key>: <value>")
        else:
            formatted = _format_type_for_toon(original_annotation, description)
            lines.append(f"{prefix}{field_name}: {formatted}")

    return "\n".join(lines)


def get_toon_system_prompt(response_model: type[Any]) -> str:
    """
    Generate the TOON system prompt for a given response model.

    Args:
        response_model: A Pydantic BaseModel class

    Returns:
        A formatted system prompt string for TOON mode
    """
    toon_structure = generate_toon_structure(response_model)
    return dedent(f"""
        Respond in TOON format inside a ```toon code block.

        Structure:
        ```toon
{toon_structure}
        ```

        Rules:
        - 2-space indentation for nesting
        - Arrays: field[N]: val1,val2,val3 where N = actual count
        - Tables: field[N,]{{col1,col2}}: with one row per line

        Value formatting:
        - <int>: whole numbers without quotes (e.g., age: 25)
        - <float>: decimal numbers without quotes (e.g., price: 19.99)
        - "<str>": quoted strings (e.g., name: "Alice", zip: "10001")
        - <bool>: true or false

        IMPORTANT: Output values only, not the type placeholders.
    """).strip()


def get_toon_reask_message(exception: Exception, previous_response: str = "") -> str:
    """
    Generate a reask message for TOON validation errors.

    Args:
        exception: The validation exception that occurred
        previous_response: Optional previous response text for context

    Returns:
        A formatted reask message string
    """
    base_message = (
        f"Validation error:\n{exception}\n\n"
        "Fix your TOON response:\n"
        "- int fields: whole numbers, no quotes (age: 25)\n"
        "- float fields: decimals, no quotes (price: 19.99)\n"
        '- str fields: quoted (name: "Alice")\n'
        "- Array [N] must match actual count\n\n"
        "Return corrected TOON in a ```toon code block."
    )

    if previous_response:
        return (
            f"Validation error:\n{exception}\n\nYour previous response:\n{previous_response}\n\n"
            + base_message.split("\n\n", 1)[1]
        )

    return base_message


def check_toon_import() -> None:
    """
    Check if toon-format package is installed.

    Raises:
        ImportError: If toon-format is not installed
    """
    try:
        import toon_format  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The 'toon-format' package is required for TOON mode. "
            "Install it with: pip install 'instructor[toon]' or pip install toon-format"
        ) from e
