"""Tests for GenAI config merging functionality.

These tests verify that config parameters like thinking_config are properly
extracted from user-provided GenerateContentConfig objects.

Related issues:
- #1966: thinking_config inside config parameter is ignored in GENAI_STRUCTURED_OUTPUTS mode
- #1953: GenAI automatic_function_calling config not passed through
- #1964: Optional is supported by generative-ai/*
"""

import pytest

# Skip if google-genai is not installed
genai = pytest.importorskip("google.genai")

from instructor.providers.gemini.utils import (
    update_genai_kwargs,
    verify_no_unions,
    map_to_gemini_function_schema,
)


def test_update_genai_kwargs_thinking_config_from_config_object():
    """Test that thinking_config inside config parameter is properly extracted.

    This tests the fix for issue #1966 where thinking_config inside the config
    parameter was silently ignored.
    """

    # Create a mock config object with thinking_config
    class MockThinkingConfig:
        def __init__(self, thinking_budget: int):
            self.thinking_budget = thinking_budget

    mock_thinking_config = MockThinkingConfig(thinking_budget=2048)

    # Create a config object with thinking_config attribute
    class MockConfig:
        def __init__(self):
            self.thinking_config = mock_thinking_config
            self.automatic_function_calling = None
            self.labels = None

    mock_config = MockConfig()

    kwargs = {"config": mock_config}
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Check that thinking_config was extracted from the config object
    assert "thinking_config" in result
    assert result["thinking_config"] == mock_thinking_config


def test_update_genai_kwargs_thinking_config_kwarg_priority():
    """Test that thinking_config as kwarg takes priority over config.thinking_config."""

    # Create a mock config object with thinking_config
    class MockThinkingConfigA:
        def __init__(self):
            self.thinking_budget = 1024

    class MockThinkingConfigB:
        def __init__(self):
            self.thinking_budget = 2048

    class MockConfig:
        def __init__(self):
            self.thinking_config = MockThinkingConfigA()
            self.automatic_function_calling = None
            self.labels = None

    mock_config = MockConfig()
    kwarg_thinking_config = MockThinkingConfigB()

    # Pass both config object and thinking_config kwarg
    kwargs = {"config": mock_config, "thinking_config": kwarg_thinking_config}
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Check that the kwarg thinking_config takes priority
    assert "thinking_config" in result
    assert result["thinking_config"] == kwarg_thinking_config
    assert result["thinking_config"].thinking_budget == 2048


def test_update_genai_kwargs_config_object_automatic_function_calling():
    """Test that automatic_function_calling is extracted from config object.

    This tests the fix for issue #1953 where automatic_function_calling
    config was not passed through.
    """

    class MockConfig:
        def __init__(self):
            self.thinking_config = None
            self.automatic_function_calling = True
            self.labels = {"key": "value"}

    mock_config = MockConfig()

    kwargs = {"config": mock_config}
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Check that automatic_function_calling was extracted
    assert "automatic_function_calling" in result
    assert result["automatic_function_calling"] is True

    # Check that labels was extracted
    assert "labels" in result
    assert result["labels"] == {"key": "value"}


def test_update_genai_kwargs_config_object_does_not_override_base():
    """Test that config object fields don't override existing base_config values."""

    class MockConfig:
        def __init__(self):
            self.thinking_config = None
            self.automatic_function_calling = True
            self.labels = {"config_key": "config_value"}

    mock_config = MockConfig()

    kwargs = {"config": mock_config}
    base_config = {"labels": {"base_key": "base_value"}}

    result = update_genai_kwargs(kwargs, base_config)

    # Check that base_config labels are preserved (not overridden)
    assert result["labels"] == {"base_key": "base_value"}


def test_update_genai_kwargs_no_config_object():
    """Test that function works normally when no config object is provided."""
    kwargs = {
        "generation_config": {
            "max_tokens": 100,
            "temperature": 0.7,
        }
    }
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Check that normal parameters still work
    assert result["max_output_tokens"] == 100
    assert result["temperature"] == 0.7


def test_update_genai_kwargs_config_object_with_no_thinking_config():
    """Test that function works when config object has no thinking_config."""

    class MockConfig:
        def __init__(self):
            self.automatic_function_calling = True
            # No thinking_config attribute

    mock_config = MockConfig()

    kwargs = {"config": mock_config}
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Should not have thinking_config
    assert "thinking_config" not in result
    # But should have automatic_function_calling
    assert "automatic_function_calling" in result
    assert result["automatic_function_calling"] is True


# Tests for issue #1964: Union type support
def test_verify_no_unions_always_returns_true():
    """Test that verify_no_unions now always returns True.

    This tests the fix for issue #1964 where Union types were incorrectly
    rejected even though Google GenAI now supports them.
    See: https://github.com/googleapis/python-genai/issues/447
    """
    # Test with a simple schema
    simple_schema = {"properties": {"name": {"type": "string"}}}
    assert verify_no_unions(simple_schema) is True

    # Test with Optional type (Union with null)
    optional_schema = {
        "properties": {"maybe_name": {"anyOf": [{"type": "string"}, {"type": "null"}]}}
    }
    assert verify_no_unions(optional_schema) is True

    # Test with Union type (int | str) - this used to fail, now should pass
    union_schema = {
        "properties": {"value": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}
    }
    assert verify_no_unions(union_schema) is True

    # Test with complex Union type - this used to fail, now should pass
    complex_union_schema = {
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {"type": "boolean"},
                ]
            }
        }
    }
    assert verify_no_unions(complex_union_schema) is True


def test_map_to_gemini_function_schema_accepts_union_types():
    """Test that map_to_gemini_function_schema accepts Union types.

    This tests the fix for issue #1964 where Union types like int | str
    were incorrectly rejected.
    """
    # Schema with Union type (int | str) - this used to raise ValueError
    schema = {
        "title": "TestModel",
        "type": "object",
        "properties": {
            "maybe_int": {"anyOf": [{"type": "integer"}, {"type": "string"}]}
        },
        "required": ["maybe_int"],
    }

    # This should not raise an error anymore
    result = map_to_gemini_function_schema(schema)
    assert result is not None
    assert "properties" in result
    assert "maybe_int" in result["properties"]
