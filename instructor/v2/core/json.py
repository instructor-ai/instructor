"""JSON extraction helpers owned by the v2 runtime."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Generator, Iterable


def extract_json_from_codeblock(content: str) -> str:
    """Extract the first JSON object- or array-like span from a text block."""
    for start_index, start_char in enumerate(content):
        if start_char not in "{[":
            continue

        end_stack = ["}" if start_char == "{" else "]"]
        in_string = False
        escape_next = False

        for end_index in range(start_index + 1, len(content)):
            char = content[end_index]

            if escape_next:
                escape_next = False
            elif char == "\\" and in_string:
                escape_next = True
            elif char == '"':
                in_string = not in_string

            if in_string:
                continue

            if char in "{[":
                end_stack.append("}" if char == "{" else "]")
                continue
            if end_stack and char == end_stack[-1]:
                end_stack.pop()
                if not end_stack:
                    candidate = content[start_index : end_index + 1]
                    try:
                        json.loads(candidate)
                    except Exception:
                        break
                    return candidate

    return content


def extract_json_from_stream(chunks: Iterable[str]) -> Generator[str, None, None]:
    """Extract JSON characters from a plain-text or fenced streaming response."""
    in_codeblock = False
    codeblock_delimiter_count = 0
    json_started = False
    in_string = False
    escape_next = False
    delimiter_stack: list[str] = []
    buffer: list[str] = []
    codeblock_buffer: list[str] = []

    for chunk in chunks:
        for char in chunk:
            if not in_codeblock and char == "`" and not (json_started and in_string):
                codeblock_buffer.append(char)
                if len(codeblock_buffer) == 3:
                    in_codeblock = True
                    codeblock_delimiter_count = 0
                    codeblock_buffer = []
                continue
            if codeblock_buffer and char != "`":
                codeblock_buffer = []

            if in_codeblock and not json_started:
                if char == "`":
                    codeblock_delimiter_count += 1
                    if codeblock_delimiter_count == 3:
                        in_codeblock = False
                        codeblock_delimiter_count = 0
                    continue
                if codeblock_delimiter_count > 0:
                    codeblock_delimiter_count = 0

                if char in "{[":
                    json_started = True
                    delimiter_stack.append("}" if char == "{" else "]")
                    buffer.append(char)
                continue

            if json_started:
                if escape_next:
                    escape_next = False
                elif char == "\\" and in_string:
                    escape_next = True
                    buffer.append(char)
                    continue
                elif char == '"':
                    in_string = not in_string

                if in_codeblock and not in_string:
                    if char == "`":
                        codeblock_delimiter_count += 1
                        if codeblock_delimiter_count == 3:
                            in_codeblock = False
                            yield from buffer
                            buffer = []
                            json_started = False
                            break
                        continue
                    if codeblock_delimiter_count > 0:
                        codeblock_delimiter_count = 0

                if not in_string:
                    if char in "{[":
                        delimiter_stack.append("}" if char == "{" else "]")
                    elif delimiter_stack and char == delimiter_stack[-1]:
                        delimiter_stack.pop()
                        if not delimiter_stack:
                            buffer.append(char)
                            yield from buffer
                            buffer = []
                            json_started = False
                            break

                buffer.append(char)
                continue

            if not in_codeblock and not json_started and char in "{[":
                json_started = True
                delimiter_stack.append("}" if char == "{" else "]")
                buffer.append(char)

    if json_started and buffer:
        yield from buffer


async def extract_json_from_stream_async(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Async counterpart to :func:`extract_json_from_stream`."""
    in_codeblock = False
    codeblock_delimiter_count = 0
    json_started = False
    in_string = False
    escape_next = False
    delimiter_stack: list[str] = []
    buffer: list[str] = []
    codeblock_buffer: list[str] = []

    async for chunk in chunks:
        for char in chunk:
            if not in_codeblock and char == "`" and not (json_started and in_string):
                codeblock_buffer.append(char)
                if len(codeblock_buffer) == 3:
                    in_codeblock = True
                    codeblock_delimiter_count = 0
                    codeblock_buffer = []
                continue
            if codeblock_buffer and char != "`":
                codeblock_buffer = []

            if in_codeblock and not json_started:
                if char == "`":
                    codeblock_delimiter_count += 1
                    if codeblock_delimiter_count == 3:
                        in_codeblock = False
                        codeblock_delimiter_count = 0
                    continue
                if codeblock_delimiter_count > 0:
                    codeblock_delimiter_count = 0

                if char in "{[":
                    json_started = True
                    delimiter_stack.append("}" if char == "{" else "]")
                    buffer.append(char)
                continue

            if json_started:
                if escape_next:
                    escape_next = False
                elif char == "\\" and in_string:
                    escape_next = True
                    buffer.append(char)
                    continue
                elif char == '"':
                    in_string = not in_string

                if in_codeblock and not in_string:
                    if char == "`":
                        codeblock_delimiter_count += 1
                        if codeblock_delimiter_count == 3:
                            in_codeblock = False
                            for buffered_char in buffer:
                                yield buffered_char
                            buffer = []
                            json_started = False
                            break
                        continue
                    if codeblock_delimiter_count > 0:
                        codeblock_delimiter_count = 0

                if not in_string:
                    if char in "{[":
                        delimiter_stack.append("}" if char == "{" else "]")
                    elif delimiter_stack and char == delimiter_stack[-1]:
                        delimiter_stack.pop()
                        if not delimiter_stack:
                            buffer.append(char)
                            for buffered_char in buffer:
                                yield buffered_char
                            buffer = []
                            json_started = False
                            break

                buffer.append(char)
                continue

            if not in_codeblock and not json_started and char in "{[":
                json_started = True
                delimiter_stack.append("}" if char == "{" else "]")
                buffer.append(char)

    if json_started and buffer:
        for buffered_char in buffer:
            yield buffered_char
