"""Backward compatibility module for instructor.multimodal imports."""

# Re-export everything from the new location
from .processing.multimodal import *  # noqa: F403, F401

# This allows `from instructor.multimodal import Image, Audio` to work
# TODO: fix this in v2
