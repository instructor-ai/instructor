"""Tests for IterableBase streaming object extraction.

Covers behavior-preservation (the optimized incremental scanner must yield
exactly what the original repeated-from-zero `get_object` algorithm yielded)
and a complexity guard (an object streamed one character at a time must be
scanned in O(n), not O(n^2)).
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import pytest
from pydantic import BaseModel

from instructor.v2.dsl.iterable import IterableBase, IterableModel


class User(BaseModel):
    name: str


Multi = IterableModel(User)


def chunked(s: str, size: int) -> list[str]:
    return [s[i : i + size] for i in range(0, len(s), size)] or [""]


def ref_stream(chunks: list[str], task_type: type[BaseModel]) -> list[BaseModel]:
    """Reference implementation mirroring the ORIGINAL tasks_from_chunks,
    built directly on the untouched public `get_object`. Used as the oracle
    for behavior-preservation."""
    started = False
    potential = ""
    out: list[BaseModel] = []
    for chunk in chunks:
        potential += chunk
        if not started and "[" in chunk:
            started = True
            potential = chunk[chunk.find("[") + 1 :]
        while True:
            task_json, potential = IterableBase.get_object(potential, 0)
            if task_json:
                out.append(task_type.model_validate_json(task_json))
            else:
                break
    return out


PAYLOADS = [
    '[{"name":"a"}]',
    '[{"name":"a"},{"name":"b"}]',
    '[{"name":"a"}, {"name":"b"}, {"name":"c"}]',
    "[]",
    '   [ {"name":"a"} ] ',
    'prefix[{"name":"a"},{"name":"b"}]',
    '[{"name":"a{b}c"}]',  # balanced braces inside a string value
]

CHUNK_SIZES = [1, 2, 3, 7, 10_000]


@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_tasks_from_chunks_matches_reference(payload: str, size: int) -> None:
    chunks = chunked(payload, size)
    got = list(Multi.tasks_from_chunks(iter(chunks)))
    expected = ref_stream(chunks, User)
    assert [u.model_dump() for u in got] == [u.model_dump() for u in expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", PAYLOADS)
async def test_tasks_from_chunks_async_matches_reference(payload: str) -> None:
    chunks = chunked(payload, 1)

    async def agen() -> AsyncGenerator[str, None]:
        for c in chunks:
            yield c

    got = [u async for u in Multi.tasks_from_chunks_async(agen())]
    expected = ref_stream(chunks, User)
    assert [u.model_dump() for u in got] == [u.model_dump() for u in expected]


RAW_CASES = [
    '{"a":1}',
    '{"a":1}, {"b":2}]',
    '{"a":{"b":2}}]',
    '{"x":',
    '   {"a":1}',
    '{"a":"x}"}',  # quirk: brace inside a string ends the object early
    '}{"a":1}',  # stray closing brace before the object
    "nobraces",
    "",
    "{}",
]


@pytest.mark.parametrize("s", RAW_CASES)
def test_scan_for_object_matches_get_object_from_scratch(s: str) -> None:
    ref = IterableBase.get_object(s, 0)
    obj, rem, _depth, _scanned = IterableBase._scan_for_object(s, 0, 0)
    assert (obj, rem) == ref


def _count_examined_chars(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch _scan_for_object with a wrapper that counts how many characters
    the scanner reads via integer indexing. Returns the live counter dict."""
    counter = {"n": 0}
    real = IterableBase.__dict__["_scan_for_object"].__func__

    class CountingStr(str):
        def __getitem__(self, key: object) -> object:
            if isinstance(key, int):
                counter["n"] += 1
            return str.__getitem__(self, key)  # type: ignore[index]

    def counting(buffer: str, depth: int, scanned: int):
        return real(CountingStr(buffer), depth, scanned)

    monkeypatch.setattr(IterableBase, "_scan_for_object", staticmethod(counting))
    return counter


def test_incomplete_object_scanned_in_linear_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = "[" + json.dumps({"name": "x" * 4000}) + "]"
    chunks = list(payload)  # one character per chunk

    counter = _count_examined_chars(monkeypatch)
    got = list(Multi.tasks_from_chunks(iter(chunks)))

    assert len(got) == 1
    assert got[0].name == "x" * 4000
    assert counter["n"] <= 4 * len(payload), (
        f"scanner examined {counter['n']} chars for {len(payload)}-char input; "
        "expected O(n). Re-scanning the whole buffer each chunk is O(n^2)."
    )


@pytest.mark.asyncio
async def test_incomplete_object_scanned_in_linear_time_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = "[" + json.dumps({"name": "y" * 4000}) + "]"
    chunks = list(payload)

    async def agen() -> AsyncGenerator[str, None]:
        for c in chunks:
            yield c

    counter = _count_examined_chars(monkeypatch)
    got = [u async for u in Multi.tasks_from_chunks_async(agen())]

    assert len(got) == 1
    assert got[0].name == "y" * 4000
    assert counter["n"] <= 4 * len(payload), (
        f"scanner examined {counter['n']} chars for {len(payload)}-char input; "
        "expected O(n)."
    )
