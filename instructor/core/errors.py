"""
Error formatting utilities for rich error messages.

This module provides helper functions to create informative error messages
with context about providers, modes, configuration, and documentation links.
"""

from __future__ import annotations

from typing import Any


def format_mode_error(
    mode: str, provider: str, valid_modes: list[str]
) -> str:
    """
    Format a mode error message with helpful context.

    Args:
        mode: The invalid mode that was attempted
        provider: The provider name
        valid_modes: List of valid modes for this provider

    Returns:
        Formatted error message string
    """
    valid_modes_str = ", ".join(sorted(valid_modes))
    docs_link = "https://github.com/instructor-ai/instructor/blob/main/docs/integrations/"
    
    return (
        f"Invalid mode '{mode}' for provider '{provider}'. "
        f"Valid modes: {valid_modes_str}\n\n"
        f"For more information, see: {docs_link}{provider.lower()}.md"
    )


def format_config_error(
    provider: str, env_var: str | None = None, missing_key: str | None = None
) -> str:
    """
    Format a configuration error message with helpful suggestions.

    Args:
        provider: The provider name
        env_var: The environment variable that should be set (optional)
        missing_key: The configuration key that is missing (optional)

    Returns:
        Formatted error message string
    """
    parts = [f"Configuration error for provider '{provider}'."]
    
    if env_var:
        parts.append(f"\nPlease set the environment variable: {env_var}")
        parts.append(f"Example: export {env_var}=your-api-key-here")
    
    if missing_key:
        parts.append(f"\nMissing required configuration: {missing_key}")
    
    parts.append(
        f"\nFor setup instructions, see: "
        f"https://github.com/instructor-ai/instructor/blob/main/docs/integrations/{provider.lower()}.md"
    )
    
    return "".join(parts)


def format_provider_error(
    provider: str, message: str, context: dict[str, Any] | None = None
) -> str:
    """
    Format a provider-specific error message with context.

    Args:
        provider: The provider name
        message: The error message
        context: Optional dictionary of additional context

    Returns:
        Formatted error message string
    """
    parts = [f"{provider}: {message}"]
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        parts.append(f"\nContext: {context_str}")
    
    parts.append(
        f"\nFor troubleshooting, see: "
        f"https://github.com/instructor-ai/instructor/blob/main/docs/integrations/{provider.lower()}.md"
    )
    
    return "".join(parts)
