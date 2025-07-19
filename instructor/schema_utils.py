"""Backward compatibility module for instructor.schema_utils imports."""

# Re-export everything from the new location
from .processing.schema import *  # noqa: F403, F401

# This allows `from instructor.schema_utils import generate_openai_schema` to work
