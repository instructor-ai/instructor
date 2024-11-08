# type: ignore[all]
from pydantic import BaseModel, Field
from instructor.dsl.partial import Partial, PartialStringHandlingMixin
import pytest
import instructor
from openai import OpenAI, AsyncOpenAI

models = ["gpt-4o-mini"]
modes = [
    instructor.Mode.TOOLS,
]


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


def test_partial_with_whitespace():
    partial = Partial[SamplePartial]
    # Handle any leading whitespace from the model
    expected_model_dicts = [
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {"b": 1}},
    ]

    for model, expected_dict in zip(
        partial.model_from_chunks(["\n", "\t", " ", '{"b": {"b": 1}}']),
        expected_model_dicts,
    ):
        assert model.model_dump() == expected_dict

    assert model.model_dump() == {"a": None, "b": {"b": 1}}


@pytest.mark.asyncio
async def test_async_partial_with_whitespace():
    partial = Partial[SamplePartial]

    # Handle any leading whitespace from the model
    async def async_generator():
        for chunk in ["\n", "\t", " ", '{"b": {"b": 1}}']:
            yield chunk

    expected_model_dicts = [
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {"b": 1}},
    ]

    i = 0
    async for model in partial.model_from_chunks_async(async_generator()):
        assert model.model_dump() == expected_model_dicts[i]
        i += 1

    assert model.model_dump() == {"a": None, "b": {"b": 1}}


def test_summary_extraction():
    class Summary(BaseModel, PartialStringHandlingMixin):
        summary: str = Field(description="A detailed summary")

    client = OpenAI()
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    extraction_stream = client.chat.completions.create_partial(
        model="gpt-4o",
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    for extraction in extraction_stream:
        if previous_summary is not None and extraction:
            updates += 1
        previous_summary = extraction.summary

    assert updates == 1


@pytest.mark.asyncio
async def test_summary_extraction_async():
    class Summary(BaseModel, PartialStringHandlingMixin):
        summary: str = Field(description="A detailed summary")

    client = AsyncOpenAI()
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    extraction_stream = client.chat.completions.create_partial(
        model="gpt-4o",
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    async for extraction in extraction_stream:
        if previous_summary is not None and extraction:
            updates += 1
        previous_summary = extraction.summary

    assert updates == 1
