# conftest.py
import os
import pytest

if not os.getenv("ANTHROPIC_API_KEY"):
    pytest.skip(
        "ANTHROPIC_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from anthropic import AsyncAnthropic, Anthropic
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("anthropic package is not installed", allow_module_level=True)
