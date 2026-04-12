"""Processing components for request/response handling."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "OpenAISchema",
    "openai_schema",
    "convert_messages",
    "handle_response_model",
    "process_response",
    "process_response_async",
    "handle_reask_kwargs",
    "generate_openai_schema",
    "generate_anthropic_schema",
    "generate_gemini_schema",
    "Validator",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "OpenAISchema": (".function_calls", "OpenAISchema"),
    "openai_schema": (".function_calls", "openai_schema"),
    "convert_messages": (".multimodal", "convert_messages"),
    "handle_response_model": (".response", "handle_response_model"),
    "process_response": (".response", "process_response"),
    "process_response_async": (".response", "process_response_async"),
    "handle_reask_kwargs": (".response", "handle_reask_kwargs"),
    "generate_openai_schema": (".schema", "generate_openai_schema"),
    "generate_anthropic_schema": (".schema", "generate_anthropic_schema"),
    "generate_gemini_schema": (".schema", "generate_gemini_schema"),
    "Validator": (".validators", "Validator"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_path, package=__name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
