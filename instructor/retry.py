"""Backward compatibility module for instructor.retry imports."""

# Re-export everything from the new location
from .core.retry import *  # noqa: F403, F401

# This allows `from instructor.retry import extract_messages` to work
