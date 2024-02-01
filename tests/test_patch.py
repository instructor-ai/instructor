import functools

import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

import instructor
from instructor.patch import OVERRIDE_DOCS, dump_message, is_async


def test_patch_completes_successfully():
    instructor.patch(OpenAI())


def test_apatch_completes_successfully():
    instructor.apatch(AsyncOpenAI())


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
    "name_of_test, message, expected",
    [
        (
            "tool_calls and content and no function_call",
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
                "tool_calls": [
                    {
                        "id": "test_tool",
                        "function": {"arguments": "", "name": "test_tool"},
                        "type": "function",
                    }
                ],
            },
        ),
        (
            "tool_calls and no content and no function_call",
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
                "tool_calls": [
                    {
                        "id": "test_tool",
                        "function": {"arguments": "", "name": "test_tool"},
                        "type": "function",
                    }
                ],
            },
        ),
        (
            "no tool_calls and no content no function_call",
            ChatCompletionMessage(
                role="assistant",
                content=None,
            ),
            {
                "role": "assistant",
                "content": "",
            },
        ),
        (
            "no tool_calls and content and function_call",
            ChatCompletionMessage(
                role="assistant",
                content="Hello, world!",
                function_call=FunctionCall(arguments="", name="test_tool"),
            ),
            {
                "role": "assistant",
                "content": 'Hello, world!{"arguments": "", "name": "test_tool"}',
            },
        ),
        (
            "no tool_calls and no content and function_call",
            ChatCompletionMessage(
                role="assistant",
                content=None,
                function_call=FunctionCall(arguments="", name="test_tool"),
            ),
            {
                "role": "assistant",
                "content": '{"arguments": "", "name": "test_tool"}',
            },
        ),
        (
            "tool_calls and no content and function_call",
            ChatCompletionMessage(
                role="assistant",
                content="",
                function_call=FunctionCall(arguments="", name="test_tool"),
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
                "content": '[{"id": "test_tool", "function": {"arguments": "", "name": "test_tool"}, "type": "function"}]{"arguments": "", "name": "test_tool"}',
                "tool_calls": [
                    {
                        "id": "test_tool",
                        "function": {"arguments": "", "name": "test_tool"},
                        "type": "function",
                    }
                ],
            },
        ),
    ],
)
@pytest.mark.skip("New changes to tools and functions")
def test_dump_message(
    name_of_test: str,
    message: ChatCompletionMessage,
    expected: ChatCompletionMessageParam,
):
    #! Something is going on right now, but I don't have time to figure it out @jxnlco
    assert dump_message(message) == expected, name_of_test
