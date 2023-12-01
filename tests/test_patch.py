import functools

import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

import instructor
from instructor.patch import OVERRIDE_DOCS, dump_message, is_async, wrap_chatcompletion


def test_patch_completes_successfully():
    instructor.patch(OpenAI())


def test_apatch_completes_successfully():
    instructor.apatch(AsyncOpenAI())


@pytest.mark.asyncio
async def test_wrap_chatcompletion_wraps_async_input_function():
    async def input_function(*args, **kwargs):
        return "Hello, World!"

    wrapped_function = wrap_chatcompletion(input_function)
    result = await wrapped_function()

    assert result == "Hello, World!"


def test_wrap_chatcompletion_wraps_input_function():
    def input_function(*args, **kwargs):
        return "Hello, World!"

    wrapped_function = wrap_chatcompletion(input_function)
    result = wrapped_function()

    assert result == "Hello, World!"


def test_is_async_returns_true_if_function_is_async():
    async def async_function():
        pass

    assert is_async(async_function) is True


def test_is_async_returns_false_if_function_is_not_async():
    def sync_function():
        pass

    assert is_async(sync_function) is False


def test_is_async_returns_true_if_wrapped_function_is_async():
    async def async_function():
        pass

    @functools.wraps(async_function)
    def wrapped_function():
        pass

    assert is_async(wrapped_function) is True


def test_override_docs():
    assert (
        "response_model" in OVERRIDE_DOCS
    ), "response_model should be in OVERRIDE_DOCS"


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            ChatCompletionMessage(
                role="assistant",
                content="Hello, world!",
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="test_tool",
                        function=Function(arguments="", name="test_tool"),
                        type="function",
                    )
                ],
            ),
            {
                "role": "assistant",
                "content": 'Hello, world![{"id": "test_tool", "function": {"arguments": "", "name": "test_tool"}, "type": "function"}]',
            },
        ),
        (
            ChatCompletionMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="test_tool",
                        function=Function(arguments="", name="test_tool"),
                        type="function",
                    )
                ],
            ),
            {
                "role": "assistant",
                "content": '[{"id": "test_tool", "function": {"arguments": "", "name": "test_tool"}, "type": "function"}]',
            },
        ),
        (
            ChatCompletionMessage(
                role="assistant",
                content=None,
            ),
            {
                "role": "assistant",
                "content": "",
            },
        ),
    ],
)
def test_dump_message(
    message: ChatCompletionMessage, expected: ChatCompletionMessageParam
):
    assert dump_message(message) == expected
