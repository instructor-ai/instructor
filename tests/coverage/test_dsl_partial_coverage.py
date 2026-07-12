from collections.abc import AsyncGenerator, Generator, Iterable
from enum import Enum
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.dsl.json_tracker import JsonCompleteness
from instructor.v2.dsl.partial import (
    Partial,
    PartialBase,
    _build_partial_list,
    _build_partial_object,
)


class State(str, Enum):
    READY = "ready"
    PENDING = "pending"


class Item(BaseModel):
    number: int
    state: State


class Envelope(BaseModel):
    optional_note: str | None
    metadata: dict[str, str | int]
    featured: Item
    items: list[Item]


class TextChunk:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class UnprintableChunk:
    def __str__(self) -> str:
        raise ValueError("cannot render chunk")


def test_partial_builder_validates_complete_nested_values_and_keeps_open_values() -> (
    None
):
    tracker = JsonCompleteness()
    tracker.analyze(
        '{"optional_note":null,"metadata":{"source":"stream","attempt":2},'
        '"featured":{"number":"7","state":"ready"},'
        '"items":[{"number":"8","state":"pending"},'
        '{"number":"9","state":"rea'
    )

    result = _build_partial_object(
        {
            "optional_note": None,
            "metadata": {"source": "stream", "attempt": 2},
            "featured": {"number": "7", "state": "ready"},
            "items": [
                {"number": "8", "state": "pending"},
                {"number": "9", "state": "rea"},
            ],
        },
        Envelope,
        tracker,
        "",
    )

    assert result.optional_note is None
    assert result.metadata == {"source": "stream", "attempt": 2}
    assert result.featured == Item(number=7, state=State.READY)
    assert result.items[0] == Item(number=8, state=State.PENDING)
    assert result.items[1] == {"number": "9", "state": "rea"}


def test_partial_builder_recurses_into_open_nested_model_and_handles_scalars() -> None:
    tracker = JsonCompleteness()
    tracker.analyze('{"featured":{"number":"12","state":"pen')

    result = _build_partial_object(
        {"featured": {"number": "12", "state": "pen"}}, Envelope, tracker, ""
    )

    assert result.featured.model_dump() == {"number": "12", "state": "pen"}
    assert result.optional_note is None
    assert result.metadata is None
    assert result.items is None
    assert _build_partial_object(None, Envelope, tracker, "") is None
    assert _build_partial_object("still streaming", Envelope, tracker, "") == (
        "still streaming"
    )


def test_partial_list_preserves_untyped_and_unknown_fields() -> None:
    tracker = JsonCompleteness()
    tracker.analyze('[{"number":"3","state":"ready"}]')
    raw_item = {"number": "3", "state": "ready"}

    assert _build_partial_list([raw_item], None, "items", tracker, "") == [raw_item]
    assert _build_partial_list([raw_item], Envelope, "missing", tracker, "") == [
        raw_item
    ]
    assert _build_partial_list([raw_item], Envelope, "metadata", tracker, "") == [
        raw_item
    ]


def test_sync_stream_accepts_string_like_chunks_and_skips_bad_chunks() -> None:
    partial_envelope = Partial[Envelope]
    values = list(
        partial_envelope.model_from_chunks(
            [
                None,
                UnprintableChunk(),
                '{"optional_note":"live","metadata":{"source":"sync"},',
                TextChunk(
                    '"featured":{"number":1,"state":"ready"},'
                    '"items":[{"number":2,"state":"pending"}]}'
                ),
            ]
        )
    )

    assert len(values) == 2
    assert values[0].optional_note == "live"
    assert values[-1] == Envelope(
        optional_note="live",
        metadata={"source": "sync"},
        featured=Item(number=1, state=State.READY),
        items=[Item(number=2, state=State.PENDING)],
    )


@pytest.mark.asyncio
async def test_async_stream_accepts_string_like_chunks_and_skips_bad_chunks() -> None:
    async def chunks() -> AsyncGenerator[Any, None]:
        yield None
        yield UnprintableChunk()
        yield '{"optional_note":"live","metadata":{"source":"async"},'
        yield TextChunk(
            '"featured":{"number":4,"state":"pending"},'
            '"items":[{"number":5,"state":"ready"}]}'
        )

    partial_envelope = Partial[Envelope]
    values = [
        value async for value in partial_envelope.model_from_chunks_async(chunks())
    ]

    assert len(values) == 2
    assert values[0].metadata == {"source": "async"}
    assert values[-1] == Envelope(
        optional_note="live",
        metadata={"source": "async"},
        featured=Item(number=4, state=State.PENDING),
        items=[Item(number=5, state=State.READY)],
    )


def test_sync_streaming_response_requires_and_uses_an_extractor() -> None:
    partial_envelope = Partial[Envelope]
    with pytest.raises(ValueError, match="stream_extractor is required"):
        list(partial_envelope.from_streaming_response([], None))

    def extractor(completion: Iterable[str]) -> Generator[str, None, None]:
        yield from completion

    values = list(
        partial_envelope.from_streaming_response(
            [
                '{"optional_note":"extracted","metadata":{},'
                '"featured":{"number":1,"state":"ready"},"items":[]}'
            ],
            extractor,
        )
    )

    assert values == [
        Envelope(
            optional_note="extracted",
            metadata={},
            featured=Item(number=1, state=State.READY),
            items=[],
        )
    ]


@pytest.mark.asyncio
async def test_async_streaming_response_requires_and_uses_an_extractor() -> None:
    partial_envelope = Partial[Envelope]

    async def empty() -> AsyncGenerator[str, None]:
        if False:
            yield ""

    with pytest.raises(ValueError, match="stream_extractor is required"):
        [
            value
            async for value in partial_envelope.from_streaming_response_async(
                empty(), None
            )
        ]

    async def completion() -> AsyncGenerator[str, None]:
        yield (
            '{"optional_note":"extracted","metadata":{},'
            '"featured":{"number":6,"state":"pending"},"items":[]}'
        )

    async def extractor(source: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        async for chunk in source:
            yield chunk

    values = [
        value
        async for value in partial_envelope.from_streaming_response_async(
            completion(), extractor
        )
    ]

    assert values == [
        Envelope(
            optional_note="extracted",
            metadata={},
            featured=Item(number=6, state=State.PENDING),
            items=[],
        )
    ]


def test_sync_json_extractor_delegates_and_rejects_a_missing_extractor() -> None:
    def extractor(completion: Iterable[str]) -> Generator[str, None, None]:
        for chunk in completion:
            yield chunk.removeprefix("payload:")

    assert list(
        PartialBase.extract_json(["payload:{", 'payload:"ok":true}'], extractor)
    ) == ["{", '"ok":true}']
    with pytest.raises(ValueError, match="stream_extractor is required"):
        list(PartialBase.extract_json([], None))


@pytest.mark.asyncio
async def test_async_json_extractor_delegates_and_rejects_a_missing_extractor() -> None:
    async def completion() -> AsyncGenerator[str, None]:
        yield "payload:{"
        yield 'payload:"ok":true}'

    async def extractor(source: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        async for chunk in source:
            yield chunk.removeprefix("payload:")

    assert [
        chunk async for chunk in PartialBase.extract_json_async(completion(), extractor)
    ] == ["{", '"ok":true}']
    with pytest.raises(ValueError, match="stream_extractor is required"):
        [chunk async for chunk in PartialBase.extract_json_async(completion(), None)]


def test_partial_rejects_direct_use_and_wraps_a_recursive_model() -> None:
    with pytest.raises(TypeError, match="Cannot instantiate abstract Partial class"):
        Partial()

    with pytest.raises(TypeError, match="Cannot subclass .*Partial"):

        class InvalidPartial(Partial):
            pass

    class RecursiveNode(BaseModel):
        child: "RecursiveNode"

    RecursiveNode.model_rebuild()
    schema = Partial[RecursiveNode].model_json_schema()

    assert schema["$defs"]["PartialRecursiveNode"]["properties"]["child"] == {
        "$ref": "#/$defs/RecursiveNode"
    }
    assert schema["$defs"]["RecursiveNode"]["properties"]["child"] == {
        "$ref": "#/$defs/RecursiveNode"
    }
