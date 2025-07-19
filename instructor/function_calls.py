"""Backward compatibility module for instructor.function_calls imports."""

# Re-export everything from the new location
from .processing.function_calls import *  # noqa: F403, F401

# This allows `from instructor.function_calls import OpenAISchema` to work
