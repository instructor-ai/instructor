"""Validation components owned by the v2 runtime."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from instructor.v2.core.errors import AsyncValidationError
from instructor.v2.validation.async_validators import (
    ASYNC_MODEL_VALIDATOR_KEY,
    ASYNC_VALIDATOR_KEY,
    AsyncValidationContext,
    async_field_validator,
    async_model_validator,
)

if TYPE_CHECKING:
    from instructor.v2.core.validators import Validator as Validator
    from instructor.v2.validation.llm_validators import (
        llm_validator as llm_validator,
        openai_moderation as openai_moderation,
    )

__all__ = [
    "AsyncValidationContext",
    "AsyncValidationError",
    "async_field_validator",
    "async_model_validator",
    "ASYNC_VALIDATOR_KEY",
    "ASYNC_MODEL_VALIDATOR_KEY",
    "Validator",
    "llm_validator",
    "openai_moderation",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Validator": ("instructor.v2.core.validators", "Validator"),
    "llm_validator": (".llm_validators", "llm_validator"),
    "openai_moderation": (".llm_validators", "openai_moderation"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_path, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
