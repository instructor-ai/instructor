"""Backwards compatibility module for instructor.dsl.validators."""

from __future__ import annotations

import warnings


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""

    warnings.warn(
        "Importing from 'instructor.dsl.validators' is deprecated and will be removed in v2.0.0. "
        "Please update your imports to use 'instructor.validation'.",
        DeprecationWarning,
        stacklevel=2,
    )

    from .. import validation

    if hasattr(validation, name):
        return getattr(validation, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
