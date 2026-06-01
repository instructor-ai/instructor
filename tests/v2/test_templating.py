import pytest
from instructor.v2.core.templating import handle_templating
from instructor.v2.core.mode import Mode


def test_handle_templating_cohere_missing_chat_history():
    """Cohere-style first-turn call with no chat_history key must not raise KeyError."""
    result = handle_templating(
        {"message": "Hello {{ name }}"},
        mode=Mode.TOOLS,
        context={"name": "World"},
    )
    assert result["message"] == "Hello World"
    assert result["chat_history"] == []
