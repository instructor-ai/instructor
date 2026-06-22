"""Tests for get_json_schema utility — ensures non-BaseModel response models
(e.g. list, list[str], dict) no longer raise AttributeError."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from instructor.v2.core.function_calls import get_json_schema


class User(BaseModel):
    name: str
    age: int


def test_basemodel_schema_unchanged() -> None:
    """BaseModel subclasses should produce the same schema as before."""
    schema = get_json_schema(User)
    expected = User.model_json_schema()
    assert schema == expected


def test_list_type_produces_array_schema() -> None:
    """Plain ``list`` should produce an array schema with empty items."""
    schema = get_json_schema(list)
    assert schema == {"type": "array", "items": {}}


def test_list_str_produces_array_schema() -> None:
    """``list[str]`` should produce an array schema with string items."""
    schema = get_json_schema(list[str])
    assert schema == {"type": "array", "items": {"type": "string"}}


def test_list_int_produces_array_schema() -> None:
    """``list[int]`` should produce an array schema with integer items."""
    schema = get_json_schema(list[int])
    assert schema == {"type": "array", "items": {"type": "integer"}}


def test_list_basemodel_produces_array_schema() -> None:
    """``list[User]`` should produce an array schema with User items."""
    schema = get_json_schema(list[User])
    assert schema["type"] == "array"
    assert schema["items"] == User.model_json_schema()


def test_dict_type_produces_object_schema() -> None:
    """Plain ``dict`` should produce an object schema."""
    schema = get_json_schema(dict)
    assert schema == {"type": "object"}


def test_dict_str_int_produces_object_schema() -> None:
    """``dict[str, int]`` should produce an object with integer additionalProperties."""
    schema = get_json_schema(dict[str, int])
    assert schema == {"type": "object", "additionalProperties": {"type": "integer"}}


def test_primitive_str_produces_string_schema() -> None:
    """``str`` should produce a string schema."""
    schema = get_json_schema(str)
    assert schema == {"type": "string"}


def test_primitive_int_produces_integer_schema() -> None:
    """``int`` should produce an integer schema."""
    schema = get_json_schema(int)
    assert schema == {"type": "integer"}


def test_primitive_float_produces_number_schema() -> None:
    """``float`` should produce a number schema."""
    schema = get_json_schema(float)
    assert schema == {"type": "number"}


def test_primitive_bool_produces_boolean_schema() -> None:
    """``bool`` should produce a boolean schema."""
    schema = get_json_schema(bool)
    assert schema == {"type": "boolean"}


def test_unknown_type_returns_empty_schema() -> None:
    """An unrecognised type should return an empty schema (safe fallback)."""
    class Custom:
        pass

    schema = get_json_schema(Custom)
    assert schema == {}
