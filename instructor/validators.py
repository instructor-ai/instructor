"""Backwards compatibility module for instructor.validators."""

from __future__ import annotations

import warnings


def __getattr__(name: str):
    """Lazy import to provide backward compatibility for validators imports."""
    warnings.warn(
        "Importing from 'instructor.validators' is deprecated and will be removed in v2.0.0. "
        "Please update your imports to use the new location:\n"
        "  from instructor.validation import llm_validator, openai_moderation",
        DeprecationWarning,
        stacklevel=2,
    )

    from . import validation

    if hasattr(validation, name):
        return getattr(validation, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
