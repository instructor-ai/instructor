"""
Model compatibility utilities for handling provider-specific limitations.

Some models from OpenAI-compatible providers don't support all features like tool calling.
This module provides utilities to detect incompatible models and suggest appropriate modes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mode import Mode

logger = logging.getLogger("instructor")

# Models that don't support tool calling (function calling)
TOOL_INCOMPATIBLE_MODELS = {
    # Moonshot AI models
    "kimi-k2-thinking",  # Issue #1925
    # Add other incompatible models here as discovered
}


def is_tool_compatible_model(model_name: str | None) -> bool:
    """
    Check if a model supports tool/function calling.

    Args:
        model_name: The model name to check (e.g., "kimi-k2-thinking")

    Returns:
        bool: True if the model supports tool calling, False otherwise
    """
    if model_name is None:
        return True  # Assume compatible if no model name provided

    # Check exact match
    if model_name in TOOL_INCOMPATIBLE_MODELS:
        return False

    # Check prefix match (for versioned models like "kimi-k2-thinking-v1")
    for incompatible_model in TOOL_INCOMPATIBLE_MODELS:
        if model_name.startswith(incompatible_model):
            return False

    return True


def get_compatible_mode(model_name: str | None, requested_mode: Mode) -> Mode:
    """
    Get a compatible mode for the given model, falling back if necessary.

    Args:
        model_name: The model name to check
        requested_mode: The mode originally requested by the user

    Returns:
        Mode: The compatible mode (may be the same as requested_mode or a fallback)
    """
    from ..mode import Mode

    # If model supports tools, use requested mode
    if is_tool_compatible_model(model_name):
        return requested_mode

    # For tool-incompatible models, map to JSON-based modes
    tool_based_modes = {
        Mode.TOOLS,
        Mode.FUNCTIONS,
        Mode.TOOLS_STRICT,
        Mode.PARALLEL_TOOLS,
    }

    if requested_mode in tool_based_modes:
        logger.warning(
            f"Model '{model_name}' does not support tool calling (mode={requested_mode.value}). "
            f"Automatically falling back to Mode.MD_JSON. "
            f"To suppress this warning, explicitly use mode=instructor.Mode.MD_JSON when patching the client."
        )
        return Mode.MD_JSON

    # For other modes, keep as requested
    return requested_mode
