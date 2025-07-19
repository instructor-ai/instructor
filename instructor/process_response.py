"""Backward compatibility module for instructor.process_response imports."""

# Re-export everything from the new location
from .processing.response import *  # noqa: F403, F401

# This allows `from instructor.process_response import handle_response_model` to work
