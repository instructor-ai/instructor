"""Tests for retry context limits feature.

This tests the max_context_messages parameter that prevents retry amplification
attacks where context grows exponentially (506x with 10 retries).

See: https://github.com/instructor-ai/instructor/issues/2056
"""

import pytest
from instructor.core.retry import truncate_context_messages


class TestTruncateContextMessages:
    """Test suite for truncate_context_messages function."""

    def test_no_truncation_when_none(self):
        """Should return kwargs unchanged when max_context_messages is None."""
        kwargs = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ]
        }
        result = truncate_context_messages(kwargs, max_context_messages=None)
        assert result == kwargs
        assert len(result["messages"]) == 3

    def test_no_truncation_when_under_limit(self):
        """Should not truncate when message count is under limit."""
        kwargs = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        result = truncate_context_messages(kwargs, max_context_messages=5)
        assert result == kwargs
        assert len(result["messages"]) == 2

    def test_truncates_to_limit(self):
        """Should truncate messages to max_context_messages."""
        kwargs = {
            "messages": [
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "msg2"},
                {"role": "user", "content": "msg3"},
                {"role": "assistant", "content": "msg4"},
                {"role": "user", "content": "msg5"},
            ]
        }
        result = truncate_context_messages(kwargs, max_context_messages=3)
        assert len(result["messages"]) == 3
        # Should keep the most recent messages
        assert result["messages"][0]["content"] == "msg3"
        assert result["messages"][1]["content"] == "msg4"
        assert result["messages"][2]["content"] == "msg5"

    def test_preserves_system_message(self):
        """Should always preserve system message and not count it towards limit."""
        kwargs = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "msg2"},
                {"role": "user", "content": "msg3"},
                {"role": "assistant", "content": "msg4"},
                {"role": "user", "content": "msg5"},
            ]
        }
        result = truncate_context_messages(kwargs, max_context_messages=3)
        # Should have system + 3 most recent = 4 total
        assert len(result["messages"]) == 4
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."
        assert result["messages"][1]["content"] == "msg3"
        assert result["messages"][2]["content"] == "msg4"
        assert result["messages"][3]["content"] == "msg5"

    def test_handles_contents_key(self):
        """Should handle Gemini/VertexAI 'contents' format."""
        kwargs = {
            "contents": [
                {"role": "user", "parts": [{"text": "msg1"}]},
                {"role": "model", "parts": [{"text": "msg2"}]},
                {"role": "user", "parts": [{"text": "msg3"}]},
            ]
        }
        result = truncate_context_messages(kwargs, max_context_messages=2)
        assert len(result["contents"]) == 2

    def test_handles_chat_history_key(self):
        """Should handle Cohere 'chat_history' format."""
        kwargs = {
            "chat_history": [
                {"role": "USER", "message": "msg1"},
                {"role": "CHATBOT", "message": "msg2"},
                {"role": "USER", "message": "msg3"},
            ]
        }
        result = truncate_context_messages(kwargs, max_context_messages=2)
        assert len(result["chat_history"]) == 2

    def test_returns_unchanged_if_no_messages(self):
        """Should return kwargs unchanged if no message key found."""
        kwargs = {"model": "gpt-4", "temperature": 0.5}
        result = truncate_context_messages(kwargs, max_context_messages=5)
        assert result == kwargs

    def test_handles_empty_messages(self):
        """Should handle empty messages list."""
        kwargs = {"messages": []}
        result = truncate_context_messages(kwargs, max_context_messages=5)
        assert result == kwargs

    def test_does_not_mutate_original(self):
        """Should not mutate the original kwargs."""
        original_messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        kwargs = {"messages": original_messages}
        result = truncate_context_messages(kwargs, max_context_messages=2)
        # Original should be unchanged
        assert len(kwargs["messages"]) == 3
        # Result should be truncated
        assert len(result["messages"]) == 2


class TestRetryAmplificationPrevention:
    """Test that max_context_messages prevents retry amplification attack."""

    def test_simulated_retry_growth(self):
        """Simulate retry amplification and verify truncation prevents growth."""
        # Initial conversation
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Extract: John is 25"},
        ]
        
        # Simulate 10 retries, each adding 2 messages
        for i in range(10):
            # Each retry adds assistant response + error message
            messages.append({"role": "assistant", "content": f"error response {i}"})
            messages.append({"role": "user", "content": f"Error: validation failed {i}"})
        
        # Without truncation: 2 + 20 = 22 messages
        assert len(messages) == 22
        
        # With truncation to 10 messages
        kwargs = {"messages": messages}
        result = truncate_context_messages(kwargs, max_context_messages=10)
        
        # Should have system + 10 recent = 11 total
        assert len(result["messages"]) == 11
        assert result["messages"][0]["role"] == "system"  # System preserved
