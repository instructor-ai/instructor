"""Regression test for #2459 — validation_context without 'context' key."""

from __future__ import annotations

from pydantic import BaseModel, Field

from instructor import CitationMixin


class User(CitationMixin, BaseModel):
    name: str = Field(description="The name of the person")


def test_validate_sources_tolerates_missing_context_key():
    user = User.model_validate(
        {"name": "Jason", "substring_quotes": ["Jason"]},
        context={"foo": "bar"},
    )
    assert user.name == "Jason"
    # Quotes are left as-is when there is no source text to resolve against.
    assert user.substring_quotes == ["Jason"]


def test_validate_sources_still_resolves_when_context_present():
    context = "Jason was a student. Jason is 20 years old."
    user = User.model_validate(
        {"name": "Jason", "substring_quotes": ["Jason was a student"]},
        context={"context": context},
    )
    assert "Jason was a student" in user.substring_quotes
