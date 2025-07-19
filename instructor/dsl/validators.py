"""Backward compatibility module for instructor.dsl.validators imports."""

# Re-export everything from the new location
from ..validation import *  # noqa: F403, F401

# This allows `from instructor.dsl.validators import llm_validator` to work
