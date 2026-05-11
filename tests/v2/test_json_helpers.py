from __future__ import annotations

import pytest

from instructor.v2.core.json import (
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
)


def test_extract_json_from_codeblock_returns_json_span() -> None:
    assert (
        extract_json_from_codeblock('prefix ```json\n{"name":"Ada"}\n``` suffix')
        == '{"name":"Ada"}'
    )


def test_extract_json_from_stream_handles_plain_json() -> None:
    assert "".join(extract_json_from_stream(["before ", '{"a":', "1}", " after"])) == (
        '{"a":1}'
    )


def test_extract_json_from_stream_handles_fenced_json() -> None:
    assert (
        "".join(extract_json_from_stream(["```json\n", '{"a":"{ok}"}', "\n```"]))
        == '{"a":"{ok}"}'
    )


@pytest.mark.asyncio
async def test_extract_json_from_stream_async_handles_fenced_json() -> None:
    async def chunks():
        for chunk in ["```json\n", '{"a":1}', "\n```"]:
            yield chunk

    assert "".join(
        [chunk async for chunk in extract_json_from_stream_async(chunks())]
    ) == ('{"a":1}')
