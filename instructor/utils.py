import json
import re
from typing import Generator, Iterable, AsyncGenerator

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)


def extract_json_from_codeblock(content: str) -> str:
    """
    Extracts the longest JSON object from a given string content that may contain one or more JSON objects.

    Args:
        content (str): The string content from which to extract the JSON object.

    Returns:
        str: The longest JSON object found within the content.

    Raises:
        ValueError: If no JSON object can be found within the content.
    """
    pattern = r"\{.*?\}"
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:
        # Return the longest JSON object found within the content
        return max(matches, key=len)
    else:
        # Raise an error if no JSON object is found
        raise ValueError(
            f"No JSON found in codeblock. Ensure the content contains a valid JSON object in a codeblock.\n{content}"
        )


def extract_json_from_stream(chunks: Iterable[str]) -> Generator[str, None, None]:
    """
    Extracts JSON objects from a stream of strings, yielding each character of the JSON objects as they are identified.

    This function iterates over a sequence of string chunks, looking for characters that denote the start ({) and end (}) of a JSON object.
    It maintains a count of open braces to ensure nested JSON objects are correctly identified and extracted in their entirety.
    Characters belonging to a JSON object are yielded one by one until the entire object has been processed.

    Args:
        chunks (Iterable[str]): A sequence of string chunks to be searched for JSON objects.

    Yields:
        Generator[str, None, None]: A generator yielding each character of the JSON objects found in the input stream.
    """
    capturing = False
    brace_count = 0
    for chunk in chunks:
        for char in chunk:
            if char == "{":
                capturing = True
                brace_count += 1
                yield char
            elif char == "}" and capturing:
                brace_count -= 1
                yield char
                if brace_count == 0:
                    capturing = False
                    break  # Cease yielding upon closing the current JSON object
            elif capturing:
                yield char


async def extract_json_from_stream_async(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """
    Asynchronously extracts JSON objects from a stream of strings, yielding each character of the JSON objects as they are identified.

    This function asynchronously iterates over a sequence of string chunks, looking for characters that denote the start ({) and end (}) of a JSON object.
    It maintains a count of open braces to ensure nested JSON objects are correctly identified and extracted in their entirety.
    Characters belonging to a JSON object are yielded one by one until the entire object has been processed.

    Args:
    chunks (AsyncGenerator[str, None]): An asynchronous sequence of string chunks to be searched for JSON objects.

    Yields:
    AsyncGenerator[str, None]: An asynchronous generator yielding each character of the JSON objects found in the input stream.
    """
    capturing = False
    brace_count = 0
    async for chunk in chunks:
        for char in chunk:
            if char == "{":
                capturing = True
                brace_count += 1
                yield char
            elif char == "}" and capturing:
                brace_count -= 1
                yield char
                if brace_count == 0:
                    capturing = False
                    break  # Cease yielding upon closing the current JSON object
            elif capturing:
                yield char


def update_total_usage(response, total_usage):
    if isinstance(response, ChatCompletion) and response.usage is not None:
        total_usage.completion_tokens += response.usage.completion_tokens or 0
        total_usage.prompt_tokens += response.usage.prompt_tokens or 0
        total_usage.total_tokens += response.usage.total_tokens or 0
        response.usage = total_usage  # Replace each response usage with the total usage
    return response


def dump_message(message: ChatCompletionMessage) -> ChatCompletionMessageParam:
    """Dumps a message to a dict, to be returned to the OpenAI API.
    Workaround for an issue with the OpenAI API, where the `tool_calls` field isn't allowed to be present in requests
    if it isn't used.
    """
    ret: ChatCompletionMessageParam = {
        "role": message.role,
        "content": message.content or "",
    }
    if hasattr(message, "tool_calls") and message.tool_calls is not None:
        ret["tool_calls"] = message.model_dump()["tool_calls"]
    if (
        hasattr(message, "function_call")
        and message.function_call is not None
        and ret["content"]
    ):
        ret["content"] += json.dumps(message.model_dump()["function_call"])
    return ret
