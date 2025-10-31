"""Deprecated shim for validator models."""

from __future__ import annotations

import warnings

from ..validation.models import Validator

warnings.warn(
    "'instructor.processing.validators' is deprecated and will be removed in a future release. "
    "Import from 'instructor.validation' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Validator"]
