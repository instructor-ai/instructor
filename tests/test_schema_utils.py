"""Tests for the new schema_utils functions."""

import pytest
from pydantic import BaseModel, Field
from typing import Optional

from instructor.processing.schema import (
    generate_openai_schema,
    generate_anthropic_schema,
    generate_gemini_schema,
)
from instructor.processing.function_calls import OpenAISchema


class TestModel(BaseModel):
    """A test model for schema generation."""

    name: str = Field(description="The name of the user")
    age: int = Field(description="The age of the user")
    email: Optional[str] = Field(default=None, description="The email address")


class TestModelWithDocstring(BaseModel):
    """A model with parameter docstring.

    Args:
        name: The full name
        age: Age in years
        tags: List of tags
    """

    name: str
    age: int
    tags: list[str] = Field(default_factory=list)


class TestModelOldStyle(TestModel, OpenAISchema):
    """Test model inheriting from OpenAISchema for comparison."""

    pass


def test_generate_openai_schema_matches_class_method():
    """Test that generate_openai_schema produces identical output to the class method."""
    # Compare with old style inheritance - but use the same model for both
    standalone_schema = generate_openai_schema(TestModelOldStyle)
    class_schema = TestModelOldStyle.openai_schema

    assert standalone_schema == class_schema

    # Test structure
    assert "name" in standalone_schema
    assert "description" in standalone_schema
    assert "parameters" in standalone_schema
    assert "properties" in standalone_schema["parameters"]
    assert "required" in standalone_schema["parameters"]


def test_generate_anthropic_schema_matches_class_method():
    """Test that generate_anthropic_schema produces identical output to the class method."""
    standalone_schema = generate_anthropic_schema(TestModelOldStyle)
    class_schema = TestModelOldStyle.anthropic_schema

    assert standalone_schema == class_schema

    # Test structure
    assert "name" in standalone_schema
    assert "description" in standalone_schema
    assert "input_schema" in standalone_schema


@pytest.mark.skipif(
    True, reason="google.generativeai not installed in test environment"
)
def test_generate_gemini_schema_matches_class_method():
    """Test that generate_gemini_schema produces identical output to the class method."""
    # This will trigger deprecation warnings, which is expected
    with pytest.warns(DeprecationWarning):
        standalone_schema = generate_gemini_schema(TestModelOldStyle)

    with pytest.warns(DeprecationWarning):
        class_schema = TestModelOldStyle.gemini_schema

    # Both should be FunctionDeclaration objects with same attributes
    assert type(standalone_schema) == type(class_schema)
    assert standalone_schema.name == class_schema.name
    assert standalone_schema.description == class_schema.description


def test_docstring_parameter_enrichment():
    """Test that docstring parameters are properly extracted."""
    schema = generate_openai_schema(TestModelWithDocstring)

    # The description should come from the docstring
    assert "parameter docstring" in schema["description"].lower()

    # Parameters should be extracted from docstring Args section
    # This is handled by docstring_parser, so we test the integration
    assert "parameters" in schema
    assert "properties" in schema["parameters"]


def test_schema_caching():
    """Test that LRU cache works correctly."""
    # Call twice and verify it's cached (same object reference)
    schema1 = generate_openai_schema(TestModel)
    schema2 = generate_openai_schema(TestModel)

    # Should be the same cached result
    assert schema1 is schema2


def test_required_fields_generation():
    """Test that required fields are correctly identified."""
    schema = generate_openai_schema(TestModel)

    # name and age are required, email is optional
    required = schema["parameters"]["required"]
    assert "name" in required
    assert "age" in required
    assert "email" not in required


def test_field_descriptions():
    """Test that field descriptions are preserved."""
    schema = generate_openai_schema(TestModel)
    properties = schema["parameters"]["properties"]

    assert properties["name"]["description"] == "The name of the user"
    assert properties["age"]["description"] == "The age of the user"
    assert properties["email"]["description"] == "The email address"


def test_schema_name_and_title():
    """Test that schema name comes from model title."""
    schema = generate_openai_schema(TestModel)

    assert schema["name"] == "TestModel"


def test_no_inheritance_required():
    """Test that models don't need to inherit from OpenAISchema."""

    # Plain Pydantic model should work
    class PlainModel(BaseModel):
        value: str

    schema = generate_openai_schema(PlainModel)

    assert schema["name"] == "PlainModel"
    assert "parameters" in schema
    assert "value" in schema["parameters"]["properties"]


def test_anthropic_schema_uses_openai_base():
    """Test that Anthropic schema reuses OpenAI schema data."""
    openai_schema = generate_openai_schema(TestModel)
    anthropic_schema = generate_anthropic_schema(TestModel)

    # Should reuse name and description from OpenAI schema
    assert anthropic_schema["name"] == openai_schema["name"]
    assert anthropic_schema["description"] == openai_schema["description"]

    # But should have its own input_schema
    assert "input_schema" in anthropic_schema
    assert anthropic_schema["input_schema"] == TestModel.model_json_schema()


if __name__ == "__main__":
    pytest.main([__file__])
