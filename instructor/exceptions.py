"""Backward compatibility module for instructor.exceptions imports."""

# Re-export everything from the new location
from .core.exceptions import *  # noqa: F403, F401

# This allows `from instructor.exceptions import InstructorRetryException` to work
