"""JSON extraction helpers owned by the v2 runtime."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator, Iterable


def extract_json_from_codeblock(content: str) -> str:
    """Extract the first JSON object- or array-like span from a text block."""
    first_brace = content.find("{")
    first_bracket = content.find("[")

    if first_brace == -1 and first_bracket == -1:
        return content

    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        last_bracket = content.rfind("]")
        if last_bracket != -1:
            return content[first_bracket : last_bracket + 1]

    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1:
        return content[first_brace : last_brace + 1]
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
            if not in_codeblock and char == "`":
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
                if char == '"' and not escape_next:
                    in_string = not in_string
                elif char == "\\" and in_string:
                    escape_next = True
                    buffer.append(char)
                    continue
                else:
                    escape_next = False

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
            if not in_codeblock and char == "`":
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
                if char == '"' and not escape_next:
                    in_string = not in_string
                elif char == "\\" and in_string:
                    escape_next = True
                    buffer.append(char)
                    continue
                else:
                    escape_next = False

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
