"""Regression tests for #2465 and #2464."""

from __future__ import annotations

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message import FunctionCall

from instructor.v2.core.messages import dump_message
from instructor.v2.providers.gemini import utils as gemini_utils


def test_update_gemini_kwargs_does_not_mutate_caller_generation_config():
    gemini_utils._default_safety_thresholds.cache_clear()
    # Avoid depending on google.genai for this pure-dict rewrite path.
    gemini_utils._default_safety_thresholds = lambda: None  # type: ignore[assignment]

    generation_config = {"max_tokens": 5, "temperature": 0.2}
    original = dict(generation_config)

    result = gemini_utils.update_gemini_kwargs(
        {"generation_config": generation_config}
    )

    assert generation_config == original
    assert result["generation_config"]["max_output_tokens"] == 5
    assert "max_tokens" not in result["generation_config"]
    assert result["generation_config"]["temperature"] == 0.2


def test_dump_message_preserves_function_call_when_content_empty():
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        function_call=FunctionCall(name="lookup", arguments='{"id":7}'),
    )
    dumped = dump_message(message)
    assert dumped["role"] == "assistant"
    assert "lookup" in dumped["content"]
    assert '"id": 7' in dumped["content"] or '"id":7' in dumped["content"]
