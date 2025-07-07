"""Test schema conversion functions for Gemini."""

import pytest
from typing import Optional
from pydantic import BaseModel
from enum import Enum

from instructor.utils import map_to_gemini_function_schema, verify_no_unions


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SimpleModel(BaseModel):
    name: str
    age: int
    is_active: bool


class OptionalModel(BaseModel):
    name: str
    age: Optional[int] = None
    description: Optional[str] = None


class EnumModel(BaseModel):
    name: str
    priority: Priority


class NestedModel(BaseModel):
    name: str
    items: list[str]
    details: SimpleModel


def test_simple_schema_conversion():
    """Test conversion strips extra pydantic fields like 'title'."""
    schema = SimpleModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)

    # Input has 'title' fields that should be stripped out
    assert "title" in schema
    assert "title" in schema["properties"]["name"]

    # Output should be clean without title fields
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean"},
        },
        "required": ["name", "age", "is_active"],
    }

    assert result == expected


def test_optional_schema_conversion():
    """Test conversion transforms anyOf[T, null] to nullable fields."""
    schema = OptionalModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)

    # Input should have anyOf with null type for optional fields
    assert schema["properties"]["age"]["anyOf"] == [
        {"type": "integer"},
        {"type": "null"},
    ]
    assert schema["properties"]["description"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]

    # Output should convert to nullable: true
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "nullable": True},
            "description": {"type": "string", "nullable": True},
        },
        "required": ["name"],
    }

    assert result == expected


def test_enum_schema_conversion():
    """Test conversion resolves $refs and adds format: enum."""
    schema = EnumModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)

    # Input should have $ref and $defs
    assert schema["properties"]["priority"]["$ref"] == "#/$defs/Priority"
    assert "$defs" in schema
    assert schema["$defs"]["Priority"]["enum"] == ["low", "medium", "high"]

    # Output should resolve the ref and add format: enum
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "format": "enum",
            },
        },
        "required": ["name", "priority"],
    }

    assert result == expected


def test_nested_schema_conversion():
    """Test conversion of schema with nested objects."""
    schema = NestedModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)

    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "items": {"type": "array", "items": {"type": "string"}},
            "details": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "is_active": {"type": "boolean"},
                },
                "required": ["name", "age", "is_active"],
            },
        },
        "required": ["name", "items", "details"],
    }

    assert result == expected


def test_verify_no_unions_valid():
    """Test verify_no_unions with valid schemas."""
    # Simple schema should pass
    simple_schema = SimpleModel.model_json_schema()
    assert verify_no_unions(simple_schema) is True

    # Optional schema should pass (Optional[T] is Union[T, None])
    optional_schema = OptionalModel.model_json_schema()
    assert verify_no_unions(optional_schema) is True


def test_verify_no_unions_invalid():
    """Test verify_no_unions with invalid union schemas."""
    # Create a schema with a true union (not just Optional)
    invalid_schema = {
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }
    assert verify_no_unions(invalid_schema) is False


def test_schema_without_refs():
    """Test schema conversion without $refs."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["name"],
    }

    result = map_to_gemini_function_schema(schema)

    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["name"],
    }

    assert result == expected


def test_schema_with_description():
    """Test schema conversion preserves descriptions."""
    schema = {
        "type": "object",
        "description": "A test object",
        "properties": {"name": {"type": "string", "description": "The name field"}},
    }

    result = map_to_gemini_function_schema(schema)

    expected = {
        "type": "object",
        "description": "A test object",
        "properties": {"name": {"type": "string", "description": "The name field"}},
    }

    assert result == expected


def test_union_type_raises_error():
    """Test that union types (except Optional) raise ValueError."""
    # Create a model with a true union type
    union_schema = {
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }

    with pytest.raises(ValueError, match="Gemini does not support Union types"):
        map_to_gemini_function_schema(union_schema)
