import re
from collections.abc import AsyncGenerator, Generator, Iterable


# Regex patterns for JSON extraction
_JSON_CODEBLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_PATTERN = re.compile(r"({[\s\S]*})")


def extract_json_from_codeblock(content: str) -> str:
    """
    Extract JSON from a string that may contain markdown code blocks or plain JSON.

    This optimized version uses regex patterns to extract JSON more efficiently.

    Args:
        content: The string that may contain JSON

    Returns:
        The extracted JSON string
    """
    # First try to find JSON in code blocks
    match = _JSON_CODEBLOCK_PATTERN.search(content)
    if match:
        json_content = match.group(1).strip()
    else:
        # Look for JSON objects with the pattern { ... }
        match = _JSON_PATTERN.search(content)
        if match:
            json_content = match.group(1)
        else:
            # Fallback to the old method if regex doesn't find anything
            first_paren = content.find("{")
            last_paren = content.rfind("}")
            if first_paren != -1 and last_paren != -1:
                json_content = content[first_paren : last_paren + 1]
            else:
                json_content = content  # Return as is if no JSON-like content found

    return json_content


def extract_json_from_stream(
    chunks: Iterable[str],
) -> Generator[str, None, None]:
    """
    Extract JSON from a stream of chunks, handling JSON in code blocks.

    This optimized version extracts JSON from markdown code blocks or plain JSON
    by implementing a state machine approach.

    The state machine tracks several states:
    - Whether we're inside a code block (```json ... ```)
    - Whether we've started tracking a JSON object
    - Whether we're inside a string literal
    - The stack of open braces to properly identify the JSON structure

    Args:
        chunks: An iterable of string chunks

    Yields:
        Characters within the JSON object
    """
    # State flags
    in_codeblock = False
    codeblock_delimiter_count = 0
    json_started = False
    in_string = False
    escape_next = False
    brace_stack = []
    buffer = []

    # Track potential codeblock start/end
    codeblock_buffer = []

    for chunk in chunks:
        for char in chunk:
            # Track codeblock delimiters (```)
            if not in_codeblock and char == "`":
                codeblock_buffer.append(char)
                if len(codeblock_buffer) == 3:
                    in_codeblock = True
                    codeblock_delimiter_count = 0
                    codeblock_buffer = []
                continue
            elif len(codeblock_buffer) > 0 and char != "`":
                # Reset if we see something other than backticks
                codeblock_buffer = []

            # If we're in a codeblock but haven't started JSON yet
            if in_codeblock and not json_started:
                # Track end of codeblock
                if char == "`":
                    codeblock_delimiter_count += 1
                    if codeblock_delimiter_count == 3:
                        in_codeblock = False
                        codeblock_delimiter_count = 0
                    continue
                elif codeblock_delimiter_count > 0:
                    codeblock_delimiter_count = (
                        0  # Reset if we see something other than backticks
                    )

                # Look for the start of JSON
                if char == "{":
                    json_started = True
                    brace_stack.append("{")
                    buffer.append(char)
                # Skip other characters until we find the start of JSON
                continue

            # If we've started tracking JSON
            if json_started:
                # Handle string literals and escaped characters
                if char == '"' and not escape_next:
                    in_string = not in_string
                elif char == "\\" and in_string:
                    escape_next = True
                    buffer.append(char)
                    continue
                else:
                    escape_next = False

                # Track end of codeblock if we're in one
                if in_codeblock and not in_string:
                    if char == "`":
                        codeblock_delimiter_count += 1
                        if codeblock_delimiter_count == 3:
                            # End of codeblock means end of JSON
                            in_codeblock = False
                            # Yield the buffer without the closing backticks
                            for c in buffer:
                                yield c
                            buffer = []
                            json_started = False
                            break
                        continue
                    elif codeblock_delimiter_count > 0:
                        codeblock_delimiter_count = 0

                # Track braces when not in a string
                if not in_string:
                    if char == "{":
                        brace_stack.append("{")
                    elif char == "}" and brace_stack:
                        brace_stack.pop()
                        # If we've completed a JSON object, yield its characters
                        if not brace_stack:
                            buffer.append(char)
                            for c in buffer:
                                yield c
                            buffer = []
                            json_started = False
                            break

                # Add character to buffer
                buffer.append(char)
                continue

            # If we're not in a codeblock and haven't started JSON, look for standalone JSON
            if not in_codeblock and not json_started and char == "{":
                json_started = True
                brace_stack.append("{")
                buffer.append(char)

    # Yield any remaining buffer content if we have valid JSON
    if json_started and buffer:
        for c in buffer:
            yield c


async def extract_json_from_stream_async(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """
    Extract JSON from an async stream of chunks, handling JSON in code blocks.

    This optimized version extracts JSON from markdown code blocks or plain JSON
    by implementing a state machine approach.

    The state machine tracks several states:
    - Whether we're inside a code block (```json ... ```)
    - Whether we've started tracking a JSON object
    - Whether we're inside a string literal
    - The stack of open braces to properly identify the JSON structure

    Args:
        chunks: An async generator yielding string chunks

    Yields:
        Characters within the JSON object
    """
    # State flags
    in_codeblock = False
    codeblock_delimiter_count = 0
    json_started = False
    in_string = False
    escape_next = False
    brace_stack = []
    buffer = []

    # Track potential codeblock start/end
    codeblock_buffer = []

    async for chunk in chunks:
        for char in chunk:
            # Track codeblock delimiters (```)
            if not in_codeblock and char == "`":
                codeblock_buffer.append(char)
                if len(codeblock_buffer) == 3:
                    in_codeblock = True
                    codeblock_delimiter_count = 0
                    codeblock_buffer = []
                continue
            elif len(codeblock_buffer) > 0 and char != "`":
                # Reset if we see something other than backticks
                codeblock_buffer = []

            # If we're in a codeblock but haven't started JSON yet
            if in_codeblock and not json_started:
                # Track end of codeblock
                if char == "`":
                    codeblock_delimiter_count += 1
                    if codeblock_delimiter_count == 3:
                        in_codeblock = False
                        codeblock_delimiter_count = 0
                    continue
                elif codeblock_delimiter_count > 0:
                    codeblock_delimiter_count = (
                        0  # Reset if we see something other than backticks
                    )

                # Look for the start of JSON
                if char == "{":
                    json_started = True
                    brace_stack.append("{")
                    buffer.append(char)
                # Skip other characters until we find the start of JSON
                continue

            # If we've started tracking JSON
            if json_started:
                # Handle string literals and escaped characters
                if char == '"' and not escape_next:
                    in_string = not in_string
                elif char == "\\" and in_string:
                    escape_next = True
                    buffer.append(char)
                    continue
                else:
                    escape_next = False

                # Track end of codeblock if we're in one
                if in_codeblock and not in_string:
                    if char == "`":
                        codeblock_delimiter_count += 1
                        if codeblock_delimiter_count == 3:
                            # End of codeblock means end of JSON
                            in_codeblock = False
                            # Yield the buffer without the closing backticks
                            for c in buffer:
                                yield c
                            buffer = []
                            json_started = False
                            break
                        continue
                    elif codeblock_delimiter_count > 0:
                        codeblock_delimiter_count = 0

                # Track braces when not in a string
                if not in_string:
                    if char == "{":
                        brace_stack.append("{")
                    elif char == "}" and brace_stack:
                        brace_stack.pop()
                        # If we've completed a JSON object, yield its characters
                        if not brace_stack:
                            buffer.append(char)
                            for c in buffer:
                                yield c
                            buffer = []
                            json_started = False
                            break

                # Add character to buffer
                buffer.append(char)
                continue

            # If we're not in a codeblock and haven't started JSON, look for standalone JSON
            if not in_codeblock and not json_started and char == "{":
                json_started = True
                brace_stack.append("{")
                buffer.append(char)

    # Yield any remaining buffer content if we have valid JSON
    if json_started and buffer:
        for c in buffer:
            yield c
