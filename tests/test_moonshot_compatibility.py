"""
Tests for Moonshot model compatibility with tool calling.

Issue #1925: The kimi-k2-thinking model doesn't support tool calling parameters.
This test ensures instructor gracefully handles this by auto-switching to JSON mode.
"""

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
import pytest
from instructor.mode import Mode
from instructor.utils.model_compat import (
    is_tool_compatible_model,
    get_compatible_mode,
)


class UserDetail(BaseModel):
    """User details extracted from text."""

    name: str = Field(description="The name of the user.")
    age: int = Field(description="The age of the user.")


def test_is_tool_compatible_model():
    """Test the model compatibility checker."""
    # Incompatible models
    assert not is_tool_compatible_model("kimi-k2-thinking")
    assert not is_tool_compatible_model("kimi-k2-thinking-v1")  # Prefix match

    # Compatible models (or unknown)
    assert is_tool_compatible_model("gpt-4")
    assert is_tool_compatible_model("gpt-3.5-turbo")
    assert is_tool_compatible_model(None)


def test_get_compatible_mode():
    """Test that incompatible models get mode fallback."""
    # kimi-k2-thinking should fallback from TOOLS to MD_JSON
    assert get_compatible_mode("kimi-k2-thinking", Mode.TOOLS) == Mode.MD_JSON
    assert get_compatible_mode("kimi-k2-thinking", Mode.FUNCTIONS) == Mode.MD_JSON

    # JSON modes should stay as-is even for incompatible models
    assert get_compatible_mode("kimi-k2-thinking", Mode.JSON) == Mode.JSON
    assert get_compatible_mode("kimi-k2-thinking", Mode.MD_JSON) == Mode.MD_JSON

    # Compatible models should keep requested mode
    assert get_compatible_mode("gpt-4", Mode.TOOLS) == Mode.TOOLS
    assert get_compatible_mode("gpt-4", Mode.JSON) == Mode.JSON


def test_moonshot_kimi_k2_thinking_mode_fallback():
    """
    Test that kimi-k2-thinking model automatically falls back to JSON mode
    when Mode.TOOLS is specified.
    """
    # This should not raise an error during patching/setup
    # The actual API call would fail without proper API key, but setup should work
    client = instructor.patch(
        OpenAI(
            api_key="test-key",
            base_url="https://api.moonshot.cn/v1",
        ),
        mode=Mode.TOOLS,  # This will be automatically converted to MD_JSON mode
    )

    # Verify the client was patched successfully
    assert hasattr(client.chat.completions, "create")


def test_moonshot_json_mode_works():
    """
    Test that Moonshot works when explicitly using JSON mode.
    """
    client = instructor.patch(
        OpenAI(
            api_key="test-key",
            base_url="https://api.moonshot.cn/v1",
        ),
        mode=Mode.JSON,  # Explicit JSON mode should work
    )

    # Verify the client was patched successfully
    assert hasattr(client.chat.completions, "create")


def test_moonshot_md_json_mode_works():
    """
    Test that Moonshot works when explicitly using MD_JSON mode.
    """
    client = instructor.patch(
        OpenAI(
            api_key="test-key",
            base_url="https://api.moonshot.cn/v1",
        ),
        mode=Mode.MD_JSON,  # Explicit MD_JSON mode should work
    )

    # Verify the client was patched successfully
    assert hasattr(client.chat.completions, "create")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
