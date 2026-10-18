"""Tests ensuring TypedDict models preserve underscore-prefixed keys."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from typing_extensions import NotRequired, Required, TypedDict

from instructor import Mode
from instructor.processing.response import handle_response_model
from instructor.v2.core.response_model import prepare_response_model


class DocumentRecord(TypedDict):
    _id: str
    name: str


class OptionalMetaRecord(TypedDict):
    _id: Required[str]
    _meta: NotRequired[str]
    title: str


class CollidingKeysRecord(TypedDict):
    _id: str
    field_id: str


def test_typeddict_underscore_keys_in_tool_parameters() -> None:
    model, request = handle_response_model(
        DocumentRecord,
        mode=Mode.TOOLS,
        messages=[{"role": "user", "content": "Extract document."}],
    )

    parameters = request["tools"][0]["function"]["parameters"]
    assert "_id" in parameters["properties"]
    assert "name" in parameters["properties"]
    assert "_id" in parameters["required"]
    assert "name" in parameters["required"]


def test_typeddict_underscore_keys_validation_and_dump() -> None:
    model = prepare_response_model(DocumentRecord)
    assert model is not None

    validated = model.model_validate({"_id": "doc-42", "name": "Summary"})
    dumped = validated.model_dump(by_alias=True)
    assert dumped == {"_id": "doc-42", "name": "Summary"}

    with pytest.raises(ValidationError):
        model.model_validate({"name": "Summary"})


def test_typeddict_underscore_keys_with_optional_fields() -> None:
    model = prepare_response_model(OptionalMetaRecord)
    assert model is not None

    validated = model.model_validate({"_id": "doc-99", "title": "Quarterly"})
    dumped = validated.model_dump(by_alias=True)
    assert dumped == {"_id": "doc-99", "_meta": None, "title": "Quarterly"}

    validated_with_meta = model.model_validate(
        {"_id": "doc-99", "_meta": "v1.2", "title": "Quarterly"}
    )
    assert validated_with_meta.model_dump(by_alias=True) == {
        "_id": "doc-99",
        "_meta": "v1.2",
        "title": "Quarterly",
    }


def test_typeddict_underscore_keys_collision_handling() -> None:
    model = prepare_response_model(CollidingKeysRecord)
    assert model is not None

    validated = model.model_validate({"_id": "primary", "field_id": "secondary"})
    dumped = validated.model_dump(by_alias=True)
    assert dumped == {"_id": "primary", "field_id": "secondary"}
