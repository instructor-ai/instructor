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


def test_extract_json_from_codeblock_preserves_array_root() -> None:
    assert (
        extract_json_from_codeblock('prefix ```json\n[{"name":"Ada"}]\n``` suffix')
        == '[{"name":"Ada"}]'
    )


def test_extract_json_from_codeblock_ignores_brackets_in_prose() -> None:
    assert (
        extract_json_from_codeblock(
            'See [section] for details.\n```json\n{"name":"Ada"}\n```\n'
        )
        == '{"name":"Ada"}'
    )


def test_extract_json_from_codeblock_handles_even_backslashes_before_quote() -> None:
    content = 'prefix {"path":"C:\\\\","name":"Ada"} suffix'

    assert extract_json_from_codeblock(content) == '{"path":"C:\\\\","name":"Ada"}'


def test_extract_json_from_stream_handles_plain_json() -> None:
    assert "".join(extract_json_from_stream(["before ", '{"a":', "1}", " after"])) == (
        '{"a":1}'
    )


def test_extract_json_from_stream_handles_fenced_json() -> None:
    assert (
        "".join(extract_json_from_stream(["```json\n", '{"a":"{ok}"}', "\n```"]))
        == '{"a":"{ok}"}'
    )


def test_extract_json_from_stream_handles_plain_array_root() -> None:
    assert (
        "".join(
            extract_json_from_stream(["before ", '[{"a":', "1},", '{"a":2}]', " after"])
        )
        == '[{"a":1},{"a":2}]'
    )


def test_extract_json_from_stream_handles_fenced_array_root() -> None:
    assert (
        "".join(extract_json_from_stream(["```json\n", '[{"a":"[ok]"}]', "\n```"]))
        == '[{"a":"[ok]"}]'
    )


def test_extract_json_from_stream_handles_even_backslashes_before_quote() -> None:
    chunks = ['prefix {"path":"C:', '\\\\","name":"Ada"} suffix']

    assert "".join(extract_json_from_stream(chunks)) == (
        '{"path":"C:\\\\","name":"Ada"}'
    )


def test_extract_json_from_stream_preserves_backticks_in_plain_string() -> None:
    chunks = ['{"md":"use ', "`", "pip install`", ' here"}']

    assert "".join(extract_json_from_stream(chunks)) == (
        '{"md":"use `pip install` here"}'
    )


def test_extract_json_from_stream_preserves_triple_backticks_in_plain_string() -> None:
    chunks = ['{"code":"', "```", "py```", '"}']

    assert "".join(extract_json_from_stream(chunks)) == '{"code":"```py```"}'


def test_extract_json_from_stream_preserves_backticks_in_fenced_string() -> None:
    chunks = ["```json\n", '{"code":"`inline`"}', "\n```"]

    assert "".join(extract_json_from_stream(chunks)) == '{"code":"`inline`"}'


@pytest.mark.asyncio
async def test_extract_json_from_stream_async_preserves_backticks_in_string() -> None:
    async def chunks():
        for chunk in ['{"md":"use ', "`", "pip install`", ' here"}']:
            yield chunk

    assert (
        "".join([chunk async for chunk in extract_json_from_stream_async(chunks())])
        == '{"md":"use `pip install` here"}'
    )


@pytest.mark.asyncio
async def test_extract_json_from_stream_async_handles_fenced_json() -> None:
    async def chunks():
        for chunk in ["```json\n", '{"a":1}', "\n```"]:
            yield chunk

    assert "".join(
        [chunk async for chunk in extract_json_from_stream_async(chunks())]
    ) == ('{"a":1}')


@pytest.mark.asyncio
async def test_extract_json_from_stream_async_handles_array_root() -> None:
    async def chunks():
        for chunk in ["prefix ", "[", '{"a":1}', "]", " suffix"]:
            yield chunk

    assert (
        "".join([chunk async for chunk in extract_json_from_stream_async(chunks())])
        == '[{"a":1}]'
    )


@pytest.mark.asyncio
async def test_extract_json_from_stream_async_handles_even_backslashes() -> None:
    async def chunks():
        for chunk in ['prefix {"path":"C:', '\\\\","name":"Ada"} suffix']:
            yield chunk

    assert (
        "".join([chunk async for chunk in extract_json_from_stream_async(chunks())])
        == '{"path":"C:\\\\","name":"Ada"}'
    )
