"""Tests for v2 message serialization helpers."""

import json

import pytest
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message import FunctionCall

from instructor.v2.core.messages import dump_message

pytestmark = pytest.mark.unit


def test_dump_message_preserves_function_call_with_empty_content() -> None:
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        function_call=FunctionCall(name="lookup", arguments='{"id":7}'),
    )

    result = dump_message(message)

    assert result["content"] == json.dumps({"arguments": '{"id":7}', "name": "lookup"})
