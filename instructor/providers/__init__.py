"""
Provider implementations for Instructor.

This package contains provider-specific implementations for different LLM providers.
"""

from instructor.providers.openai import from_openai

__all__ = [
    "from_openai",
]
