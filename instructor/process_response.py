"""Backwards compatibility module for instructor.process_response.

This module provides lazy imports to maintain backwards compatibility.
"""

import warnings


def __getattr__(name: str):
    """Lazy import to provide backward compatibility for process_response imports."""
    warnings.warn(
        f"Importing from 'instructor.process_response' is deprecated and will be removed in v2.0.0. "
        f"Please update your imports to use 'instructor.processing.response.{name}' instead:\n"
        "  from instructor.processing.response import process_response",
        DeprecationWarning,
        stacklevel=2,
    )

    from .processing import response as processing_response

    # Try to get the attribute from the processing.response module
    if hasattr(processing_response, name):
        return getattr(processing_response, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
