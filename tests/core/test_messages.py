"""Unit tests for merge_consecutive_messages protocol-field preservation."""

from instructor.v2.core.messages import merge_consecutive_messages


def test_preserves_tool_call_fields():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "result", "tool_call_id": "call_1"},
    ]

    result = merge_consecutive_messages(messages)

    assert result[0]["tool_calls"] == messages[0]["tool_calls"]
    assert result[1]["tool_call_id"] == "call_1"


def test_does_not_merge_across_tool_boundaries():
    messages = [
        {"role": "assistant", "content": "a", "tool_calls": [{"id": "call_1"}]},
        {"role": "assistant", "content": "b"},
    ]

    result = merge_consecutive_messages(messages)

    assert len(result) == 2
    assert result[0]["content"] != result[1]["content"]
    assert "tool_calls" in result[0]


def test_still_merges_plain_adjacent_messages():
    messages = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]

    result = merge_consecutive_messages(messages)

    assert len(result) == 1
    assert result[0]["content"] == "a\n\nb"


def test_preserves_name_and_function_call():
    messages = [
        {"role": "assistant", "content": "a", "name": "helper"},
        {"role": "assistant", "content": "b", "function_call": {"name": "f"}},
    ]

    result = merge_consecutive_messages(messages)

    assert len(result) == 2
    assert result[0]["name"] == "helper"
    assert result[1]["function_call"] == {"name": "f"}
