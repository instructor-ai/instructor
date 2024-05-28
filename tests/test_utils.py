import json
import pytest
from instructor.utils import (
    classproperty,
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
    merge_consecutive_messages,
)


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
