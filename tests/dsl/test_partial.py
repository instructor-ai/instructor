# type: ignore[all]
from pydantic import BaseModel

from instructor.dsl.partial import Partial


class SampleNestedPartial(BaseModel):
    b: int


class SamplePartial(BaseModel):
    a: int
    b: SampleNestedPartial


def test_partial():
    partial = Partial[SamplePartial]
    assert partial.model_json_schema() == {
        "$defs": {
            "PartialSampleNestedPartial": {
                "properties": {"b": {"title": "B", "type": "integer"}},
                "required": ["b"],
                "title": "PartialSampleNestedPartial",
                "type": "object",
            }
        },
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"$ref": "#/$defs/PartialSampleNestedPartial"},
        },
        "required": ["a", "b"],
        "title": "PartialSamplePartial",
        "type": "object",
    }, "Wrapped model JSON schema has changed"
    assert partial.get_partial_model().model_json_schema() == {
        "$defs": {
            "PartialSampleNestedPartial": {
                "properties": {
                    "b": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "title": "B",
                    }
                },
                "title": "PartialSampleNestedPartial",
                "type": "object",
            }
        },
        "properties": {
            "a": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "title": "A",
            },
            "b": {
                "anyOf": [
                    {"$ref": "#/$defs/PartialSampleNestedPartial"},
                    {"type": "null"},
                ],
                "default": {},
            },
        },
        "title": "PartialSamplePartial",
        "type": "object",
    }, "Partial model JSON schema has changed"

    for model in partial.model_from_chunks(['{"b": {"b": 1}}']):
        assert model.model_dump() == {"a": None, "b": {"b": 1}}
