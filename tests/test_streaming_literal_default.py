"""Tests for streaming Partial with Literal default values.

Ensures that single-value Literal fields with matching defaults are
pre-filled in partial responses during streaming, rather than being
set to None until the LLM returns them. (fixes #2054)
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from instructor.dsl.partial import (
    _build_partial_object,
    _has_deterministic_default,
    JsonCompleteness,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Person(BaseModel):
    type: Literal["Person"] = "Person"
    name: str
    age: int


class MultiLiteral(BaseModel):
    """Literal with multiple values should NOT be pre-filled."""

    kind: Literal["cat", "dog"] = "cat"
    name: str


class NoDefault(BaseModel):
    """Literal without default should NOT be pre-filled."""

    kind: Literal["item"]
    name: str


class NestedModel(BaseModel):
    type: Literal["wrapper"] = "wrapper"
    person: Person


# ---------------------------------------------------------------------------
# _has_deterministic_default tests
# ---------------------------------------------------------------------------


class TestHasDeterministicDefault:
    def test_single_literal_with_matching_default(self):
        field_info = Person.model_fields["type"]
        assert _has_deterministic_default(field_info) is True

    def test_multi_literal_returns_false(self):
        field_info = MultiLiteral.model_fields["kind"]
        assert _has_deterministic_default(field_info) is False

    def test_no_default_returns_false(self):
        field_info = NoDefault.model_fields["kind"]
        assert _has_deterministic_default(field_info) is False

    def test_regular_str_field_returns_false(self):
        field_info = Person.model_fields["name"]
        assert _has_deterministic_default(field_info) is False

    def test_int_field_returns_false(self):
        field_info = Person.model_fields["age"]
        assert _has_deterministic_default(field_info) is False


# ---------------------------------------------------------------------------
# _build_partial_object tests
# ---------------------------------------------------------------------------


class TestBuildPartialObjectLiteralDefault:
    def test_missing_literal_field_uses_default(self):
        """When 'type' is not in the parsed JSON, it should use the default."""
        tracker = JsonCompleteness()
        result = _build_partial_object(
            {"name": "Alice"},
            Person,
            tracker,
            "",
        )
        assert result.type == "Person"
        assert result.name == "Alice"
        assert result.age is None

    def test_present_literal_field_not_overridden(self):
        """When 'type' IS in the parsed JSON, use the parsed value."""
        tracker = JsonCompleteness()
        result = _build_partial_object(
            {"type": "Person", "name": "Bob", "age": 30},
            Person,
            tracker,
            "",
        )
        assert result.type == "Person"
        assert result.name == "Bob"
        assert result.age == 30

    def test_multi_literal_field_stays_none(self):
        """Multi-value Literal should NOT be pre-filled."""
        tracker = JsonCompleteness()
        result = _build_partial_object(
            {"name": "Whiskers"},
            MultiLiteral,
            tracker,
            "",
        )
        assert result.kind is None
        assert result.name == "Whiskers"

    def test_empty_data_fills_literal_default(self):
        """Even with no data, Literal default should appear."""
        tracker = JsonCompleteness()
        result = _build_partial_object(
            {},
            Person,
            tracker,
            "",
        )
        assert result.type == "Person"
        assert result.name is None
        assert result.age is None

    def test_nested_model_literal_defaults(self):
        """Literal defaults should work in nested models too."""
        tracker = JsonCompleteness()
        result = _build_partial_object(
            {"person": {"name": "Carol"}},
            NestedModel,
            tracker,
            "",
        )
        assert result.type == "wrapper"
        assert result.person.type == "Person"
        assert result.person.name == "Carol"

    def test_no_default_literal_stays_none(self):
        """Literal without default should remain None when missing."""
        tracker = JsonCompleteness()
        result = _build_partial_object(
            {"name": "Dave"},
            NoDefault,
            tracker,
            "",
        )
        assert result.kind is None
        assert result.name == "Dave"
