"""Tests for YAML mode functionality."""

import pytest
from pydantic import BaseModel
from instructor.mode import Mode
from instructor.processing.function_calls import openai_schema


class SimpleModel(BaseModel):
    """Simple test model."""

    name: str
    age: int


class NestedModel(BaseModel):
    """Nested test model."""

    name: str
    items: list[str]
    metadata: dict[str, str]


def test_yaml_mode_exists():
    """Test that YAML mode is available."""
    assert hasattr(Mode, "MD_YAML")
    assert Mode.MD_YAML.value == "md_yaml_mode"


def test_yaml_mode_handler():
    """Test that YAML mode handler is registered."""
    from instructor.providers.openai.utils import OPENAI_HANDLERS

    assert Mode.MD_YAML in OPENAI_HANDLERS
    assert "reask" in OPENAI_HANDLERS[Mode.MD_YAML]
    assert "response" in OPENAI_HANDLERS[Mode.MD_YAML]


def test_yaml_parsing_simple():
    """Test YAML parsing with simple model."""
    from unittest.mock import Mock

    # Wrap the model to get OpenAISchema functionality
    WrappedModel = openai_schema(SimpleModel)

    # Create a mock completion with YAML response
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = """
name: John Smith
age: 35
"""
    mock_completion.choices[0].finish_reason = "stop"

    # Parse YAML using the model's from_response method
    result = WrappedModel.from_response(
        mock_completion,
        mode=Mode.MD_YAML,
    )

    assert result.name == "John Smith"
    assert result.age == 35


def test_yaml_parsing_with_codeblock():
    """Test YAML parsing when wrapped in code block."""
    from unittest.mock import Mock

    WrappedModel = openai_schema(SimpleModel)

    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = """
```yaml
name: Alice Johnson
age: 28
```
"""
    mock_completion.choices[0].finish_reason = "stop"

    result = WrappedModel.from_response(
        mock_completion,
        mode=Mode.MD_YAML,
    )

    assert result.name == "Alice Johnson"
    assert result.age == 28


def test_yaml_parsing_nested():
    """Test YAML parsing with nested structures."""
    from unittest.mock import Mock

    WrappedModel = openai_schema(NestedModel)

    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = """
name: Project Alpha
items:
  - item1
  - item2
  - item3
metadata:
  author: John Doe
  version: "1.0"
"""
    mock_completion.choices[0].finish_reason = "stop"

    result = WrappedModel.from_response(
        mock_completion,
        mode=Mode.MD_YAML,
    )

    assert result.name == "Project Alpha"
    assert result.items == ["item1", "item2", "item3"]
    assert result.metadata["author"] == "John Doe"
    assert result.metadata["version"] == "1.0"


def test_yaml_invalid_response():
    """Test YAML parsing with invalid YAML."""
    from unittest.mock import Mock

    WrappedModel = openai_schema(SimpleModel)

    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = """
invalid: yaml: content:
  - this is
    - not valid
"""
    mock_completion.choices[0].finish_reason = "stop"

    with pytest.raises(ValueError, match="Failed to parse YAML"):
        WrappedModel.from_response(
            mock_completion,
            mode=Mode.MD_YAML,
        )
