"""Test for streaming reask bug fix.

Bug: When using streaming mode with max_retries > 1, if validation fails,
the reask handlers crash with "'Stream' object has no attribute 'choices'"
because they expect a ChatCompletion but receive a Stream object.

GitHub Issue: https://github.com/jxnl/instructor/issues/1991
"""

import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError, BaseModel, field_validator

from instructor.mode import Mode
from instructor.processing.response import handle_reask_kwargs


class MockStream:
    """Mock Stream object that mimics openai.Stream behavior."""

    def __iter__(self):
        return iter([])

    def __next__(self):
        raise StopIteration


def create_mock_validation_error():
    """Create a real Pydantic ValidationError for testing."""

    class TestModel(BaseModel):
        name: str

        @field_validator("name")
        @classmethod
        def must_have_space(cls, v):
            if " " not in v:
                raise ValueError("must contain space")
            return v

    try:
        TestModel(name="John")
    except ValidationError as e:
        return e


class TestStreamingReaskBug:
    """Tests for the streaming reask bug fix."""

    def test_reask_tools_with_stream_object_does_not_crash(self):
        """Test that reask_tools handles Stream objects without crashing.

        Previously, this would crash with:
        "'Stream' object has no attribute 'choices'"
        """
        mock_stream = MockStream()
        kwargs = {
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"type": "function", "function": {"name": "test"}}],
        }
        exception = create_mock_validation_error()

        # This should not raise an AttributeError
        result = handle_reask_kwargs(
            kwargs=kwargs,
            mode=Mode.TOOLS,
            response=mock_stream,
            exception=exception,
        )

        # Should return modified kwargs with error message
        assert "messages" in result
        assert len(result["messages"]) > 1  # Original + error message

    def test_reask_anthropic_tools_with_stream_object(self):
        """Test that Anthropic reask handler handles Stream objects."""
        mock_stream = MockStream()
        kwargs = {
            "messages": [{"role": "user", "content": "test"}],
        }
        exception = create_mock_validation_error()

        result = handle_reask_kwargs(
            kwargs=kwargs,
            mode=Mode.ANTHROPIC_TOOLS,
            response=mock_stream,
            exception=exception,
        )

        assert "messages" in result

    def test_reask_with_none_response(self):
        """Test that reask handlers handle None response gracefully."""
        kwargs = {
            "messages": [{"role": "user", "content": "test"}],
        }
        exception = create_mock_validation_error()

        result = handle_reask_kwargs(
            kwargs=kwargs,
            mode=Mode.TOOLS,
            response=None,
            exception=exception,
        )

        assert "messages" in result

    def test_reask_md_json_with_stream_object(self):
        """Test that MD_JSON reask handler handles Stream objects."""
        mock_stream = MockStream()
        kwargs = {
            "messages": [{"role": "user", "content": "test"}],
        }
        exception = create_mock_validation_error()

        result = handle_reask_kwargs(
            kwargs=kwargs,
            mode=Mode.MD_JSON,
            response=mock_stream,
            exception=exception,
        )

        assert "messages" in result


@pytest.mark.skipif(
    not pytest.importorskip("openai", reason="openai not installed"),
    reason="openai not installed",
)
class TestStreamingReaskIntegration:
    """Integration tests that require OpenAI API key."""

    @pytest.fixture
    def client(self):
        """Create instructor client if API key available."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        import instructor
        from openai import OpenAI

        return instructor.from_openai(OpenAI())

    def test_streaming_with_retries_and_failing_validator(self, client):
        """Test that streaming with retries doesn't crash on validation failure."""

        class StrictUser(BaseModel):
            name: str
            age: int

            @field_validator("name")
            @classmethod
            def name_must_have_space(cls, v: str) -> str:
                if v and " " not in v:
                    raise ValueError("Name must have first and last name")
                return v

        # This should not crash with AttributeError
        # It may raise InstructorRetryException after retries exhausted, which is expected
        from instructor.core.exceptions import InstructorRetryException

        with pytest.raises(InstructorRetryException):
            list(
                client.chat.completions.create_partial(
                    model="gpt-4o-mini",
                    max_retries=2,
                    messages=[
                        {
                            "role": "user",
                            "content": "Extract: John is 25. Return name='John' (no last name).",
                        }
                    ],
                    response_model=StrictUser,
                )
            )
