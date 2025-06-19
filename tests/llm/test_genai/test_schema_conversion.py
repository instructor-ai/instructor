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
    """Test conversion of a simple schema."""
    schema = SimpleModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)
    
    # Check basic structure
    assert "type" in result
    assert "properties" in result
    assert result["type"] == "object"
    
    # Check properties
    properties = result["properties"]
    assert "name" in properties
    assert "age" in properties
    assert "is_active" in properties
    
    # Check property types
    assert properties["name"]["type"] == "string"
    assert properties["age"]["type"] == "integer"
    assert properties["is_active"]["type"] == "boolean"


def test_optional_schema_conversion():
    """Test conversion of schema with optional fields."""
    schema = OptionalModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)
    
    properties = result["properties"]
    
    # Required field should not be nullable
    assert properties["name"]["type"] == "string"
    assert properties["name"].get("nullable") is None
    
    # Optional fields should be nullable
    assert properties["age"]["type"] == "integer" 
    assert properties["age"]["nullable"] is True
    assert properties["description"]["type"] == "string"
    assert properties["description"]["nullable"] is True


def test_enum_schema_conversion():
    """Test conversion of schema with enum fields."""
    schema = EnumModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)
    
    properties = result["properties"]
    priority_prop = properties["priority"]
    
    # Check enum handling
    assert priority_prop["type"] == "string"
    assert "enum" in priority_prop
    assert priority_prop["format"] == "enum"
    assert set(priority_prop["enum"]) == {"low", "medium", "high"}


def test_nested_schema_conversion():
    """Test conversion of schema with nested objects."""
    schema = NestedModel.model_json_schema()
    result = map_to_gemini_function_schema(schema)
    
    properties = result["properties"]
    
    # Check array field
    assert properties["items"]["type"] == "array"
    assert properties["items"]["items"]["type"] == "string"
    
    # Check nested object
    details_prop = properties["details"]
    assert details_prop["type"] == "object"
    assert "properties" in details_prop
    
    nested_props = details_prop["properties"]
    assert "name" in nested_props
    assert "age" in nested_props
    assert "is_active" in nested_props


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
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"}
                ]
            }
        }
    }
    assert verify_no_unions(invalid_schema) is False


def test_schema_without_refs():
    """Test schema conversion without $refs."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"}
        },
        "required": ["name"]
    }
    
    result = map_to_gemini_function_schema(schema)
    
    assert result["type"] == "object"
    assert "properties" in result
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["count"]["type"] == "integer"
    assert result["required"] == ["name"]


def test_schema_with_description():
    """Test schema conversion preserves descriptions."""
    schema = {
        "type": "object",
        "description": "A test object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name field"
            }
        }
    }
    
    result = map_to_gemini_function_schema(schema)
    
    assert result["description"] == "A test object"
    assert result["properties"]["name"]["description"] == "The name field"


def test_union_type_raises_error():
    """Test that union types (except Optional) raise ValueError."""
    # Create a model with a true union type
    union_schema = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"}
                ]
            }
        }
    }
    
    with pytest.raises(ValueError, match="Gemini does not support Union types"):
        map_to_gemini_function_schema(union_schema)
