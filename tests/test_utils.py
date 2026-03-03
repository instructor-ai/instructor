import json
import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice, ChatCompletionMessage
from instructor.utils import (
    classproperty,
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
    merge_consecutive_messages,
    extract_system_messages,
    combine_system_messages,
)
from instructor.utils.core import update_total_usage


def test_extract_json_from_codeblock():
    example = """
    Here is a response

    ```json
    {
        "key": "value"
    }    
    ```
    """
    result = extract_json_from_codeblock(example)
    assert json.loads(result) == {"key": "value"}


def test_extract_json_from_codeblock_no_end():
    example = """
    Here is a response

    ```json
    {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}]
    }  
    """
    result = extract_json_from_codeblock(example)
    assert json.loads(result) == {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}],
    }


def test_extract_json_from_codeblock_no_start():
    example = """
    Here is a response

    {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}, {"key": "value"}]
    }
    """
    result = extract_json_from_codeblock(example)
    assert json.loads(result) == {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}, {"key": "value"}],
    }


def test_stream_json():
    text = """here is the json for you! 
    
    ```json
    , here
    {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}]
    }
    ```

    What do you think?
    """

    def batch_strings(chunks, n=2):
        batch = ""
        for chunk in chunks:
            for char in chunk:
                batch += char
                if len(batch) == n:
                    yield batch
                    batch = ""
        if batch:  # Yield any remaining characters in the last batch
            yield batch

    result = json.loads(
        "".join(list(extract_json_from_stream(batch_strings(text, n=3))))
    )
    assert result == {"key": "value", "another_key": [{"key": {"key": "value"}}]}


@pytest.mark.asyncio
async def test_stream_json_async():
    text = """here is the json for you! 
    
    ```json
    , here
    {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}, {"key": "value"}]
    }
    ```

    What do you think?
    """

    async def batch_strings_async(chunks, n=2):
        batch = ""
        for chunk in chunks:
            for char in chunk:
                batch += char
                if len(batch) == n:
                    yield batch
                    batch = ""
        if batch:  # Yield any remaining characters in the last batch
            yield batch

    result = json.loads(
        "".join(
            [
                chunk
                async for chunk in extract_json_from_stream_async(
                    batch_strings_async(text, n=3)
                )
            ]
        )
    )
    assert result == {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}, {"key": "value"}],
    }


def test_merge_consecutive_messages():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you"},
        {"role": "assistant", "content": "Hello"},
        {"role": "assistant", "content": "I am good"},
    ]
    result = merge_consecutive_messages(messages)
    assert result == [
        {
            "role": "user",
            "content": "Hello\n\nHow are you",
        },
        {
            "role": "assistant",
            "content": "Hello\n\nI am good",
        },
    ]


def test_merge_consecutive_messages_empty():
    messages = []
    result = merge_consecutive_messages(messages)
    assert result == []


def test_merge_consecutive_messages_single():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello"},
    ]
    result = merge_consecutive_messages(messages)
    assert result == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_classproperty():
    """Test custom `classproperty` descriptor."""

    class MyClass:
        @classproperty
        def my_property(cls):
            return cls

    assert MyClass.my_property is MyClass

    class MyClass:
        clvar = 1

        @classproperty
        def my_property(cls):
            return cls.clvar

    assert MyClass.my_property == 1


def test_combine_system_messages_string_string():
    existing = "Existing message"
    new = "New message"
    result = combine_system_messages(existing, new)
    assert result == "Existing message\n\nNew message"


def test_combine_system_messages_list_list():
    existing = [{"type": "text", "text": "Existing"}]
    new = [{"type": "text", "text": "New"}]
    result = combine_system_messages(existing, new)
    assert result == [
        {"type": "text", "text": "Existing"},
        {"type": "text", "text": "New"},
    ]


def test_combine_system_messages_string_list():
    existing = "Existing"
    new = [{"type": "text", "text": "New"}]
    result = combine_system_messages(existing, new)
    assert result == [
        {"type": "text", "text": "Existing"},
        {"type": "text", "text": "New"},
    ]


def test_combine_system_messages_list_string():
    existing = [{"type": "text", "text": "Existing"}]
    new = "New"
    result = combine_system_messages(existing, new)
    assert result == [
        {"type": "text", "text": "Existing"},
        {"type": "text", "text": "New"},
    ]


def test_combine_system_messages_none_string():
    existing = None
    new = "New"
    result = combine_system_messages(existing, new)
    assert result == "New"


def test_combine_system_messages_none_list():
    existing = None
    new = [{"type": "text", "text": "New"}]
    result = combine_system_messages(existing, new)
    assert result == [{"type": "text", "text": "New"}]


def test_combine_system_messages_invalid_type():
    with pytest.raises(ValueError):
        combine_system_messages(123, "New")


def test_extract_system_messages():
    messages = [
        {"role": "system", "content": "System message 1"},
        {"role": "user", "content": "User message"},
        {"role": "system", "content": "System message 2"},
    ]
    result = extract_system_messages(messages)
    expected = [
        {"type": "text", "text": "System message 1"},
        {"type": "text", "text": "System message 2"},
    ]
    assert result == expected


def test_extract_system_messages_no_system():
    messages = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]
    result = extract_system_messages(messages)
    assert result == []


def test_combine_system_messages_with_cache_control():
    existing = [
        {
            "type": "text",
            "text": "You are an AI assistant.",
        },
        {
            "type": "text",
            "text": "This is some context.",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    new = "Provide insightful analysis."
    result = combine_system_messages(existing, new)
    expected = [
        {
            "type": "text",
            "text": "You are an AI assistant.",
        },
        {
            "type": "text",
            "text": "This is some context.",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "Provide insightful analysis."},
    ]
    assert result == expected


def test_combine_system_messages_string_to_cache_control():
    existing = "You are an AI assistant."
    new = [
        {
            "type": "text",
            "text": "Analyze this text:",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "<long text content>"},
    ]
    result = combine_system_messages(existing, new)
    expected = [
        {"type": "text", "text": "You are an AI assistant."},
        {
            "type": "text",
            "text": "Analyze this text:",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "<long text content>"},
    ]
    assert result == expected


def test_extract_system_messages_with_cache_control():
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this text:",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": "User message"},
        {"role": "system", "content": "<long text content>"},
    ]
    result = extract_system_messages(messages)
    expected = [
        {"type": "text", "text": "You are an AI assistant."},
        {
            "type": "text",
            "text": "Analyze this text:",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "<long text content>"},
    ]
    assert result == expected


def test_combine_system_messages_preserve_cache_control():
    existing = [
        {
            "type": "text",
            "text": "You are an AI assistant.",
        },
        {
            "type": "text",
            "text": "This is some context.",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    new = [
        {
            "type": "text",
            "text": "Additional instruction.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    result = combine_system_messages(existing, new)
    expected = [
        {
            "type": "text",
            "text": "You are an AI assistant.",
        },
        {
            "type": "text",
            "text": "This is some context.",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": "Additional instruction.",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    assert result == expected


class CustomUsage(CompletionUsage):
    """Simulates a provider-specific Usage subclass (e.g. litellm's Usage)."""

    def get(self, key: str, default=None):
        """Provider-specific method that would be lost if type is replaced."""
        return getattr(self, key, default)


def _make_chat_completion(usage):
    """Helper to create a ChatCompletion with given usage."""
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="test"),
            )
        ],
        created=0,
        model="test",
        object="chat.completion",
        usage=usage,
    )


def test_update_total_usage_preserves_usage_subclass_type():
    """Test that update_total_usage preserves the original usage object type.

    When response.usage is a subclass of CompletionUsage (e.g. litellm's Usage),
    the function should copy accumulated totals back to the original object
    instead of replacing it, preserving provider-specific methods. Fixes #2113.
    """
    custom_usage = CustomUsage(
        completion_tokens=10,
        prompt_tokens=20,
        total_tokens=30,
    )
    response = _make_chat_completion(custom_usage)
    total_usage = CompletionUsage(
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=0,
    )

    result = update_total_usage(response, total_usage)

    assert result is not None
    # The usage type must be preserved (not replaced with plain CompletionUsage)
    assert type(result.usage) is CustomUsage
    # Provider-specific methods should still work
    assert result.usage.get("completion_tokens") == 10
    # Token counts should be accumulated correctly
    assert result.usage.completion_tokens == 10
    assert result.usage.prompt_tokens == 20
    assert result.usage.total_tokens == 30


def test_update_total_usage_accumulates_across_retries():
    """Test that totals accumulate across multiple calls (simulating retries)."""
    total_usage = CompletionUsage(
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=0,
    )

    # First retry
    r1 = _make_chat_completion(
        CustomUsage(completion_tokens=5, prompt_tokens=10, total_tokens=15)
    )
    update_total_usage(r1, total_usage)

    # Second retry
    r2 = _make_chat_completion(
        CustomUsage(completion_tokens=7, prompt_tokens=12, total_tokens=19)
    )
    result = update_total_usage(r2, total_usage)

    assert result is not None
    assert type(result.usage) is CustomUsage
    assert result.usage.completion_tokens == 5 + 7
    assert result.usage.prompt_tokens == 10 + 12
    assert result.usage.total_tokens == 15 + 19
