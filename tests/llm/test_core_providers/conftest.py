"""
Configuration for core provider tests (OpenAI, Anthropic, Google).
"""

import os
import sys

# Add parent directory to path for shared_config import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared_config import pytest_generate_tests, pytest_configure  # noqa: E402, F401
