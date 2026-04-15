"""
Test that IncompleteOutputException is raised directly and not wrapped
in InstructorRetryException.

This is a regression test for issue #2273.
"""

import pytest
from unittest.mock import Mock

import instructor
from instructor.core.exceptions import (
    IncompleteOutputException,
    InstructorRetryException,
)
from instructor.mode import Mode
from pydantic import BaseModel


class Report(BaseModel):
    content: str


def _make_truncated_response() -> Mock:
    """Create a mock response that simulates a max_tokens truncation."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = '{"content": "partial...'
    mock_response.choices[0].finish_reason = "length"  # key: truncated
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.refusal = None
    mock_response.usage = None
    return mock_response


def test_incomplete_output_exception_not_wrapped_in_retry():
    """IncompleteOutputException should be catchable directly, not wrapped.

    Before the fix, IncompleteOutputException was caught by the generic
    Exception handler in retry_sync, forwarded to tenacity, and then
    re-raised as InstructorRetryException.  Users could not write:

        except IncompleteOutputException as e: ...

    because it was always wrapped.
    """
    mock_response = _make_truncated_response()

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(IncompleteOutputException) as exc_info:
        client.chat.completions.create(
            model="gpt-4",
            response_model=Report,
            messages=[{"role": "user", "content": "Write a long report..."}],
            max_retries=0,
        )

    assert exc_info.value.last_completion is mock_response


def test_incomplete_output_exception_catchable_with_retries():
    """IncompleteOutputException should be raised directly even with retries > 0.

    Retrying on a max_tokens truncation is pointless because the same
    token limit will apply.  The exception should escape immediately.
    """
    mock_response = _make_truncated_response()

    call_count = 0
    original_create = Mock(return_value=mock_response)

    def counting_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_create(*args, **kwargs)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = counting_create

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(IncompleteOutputException):
        client.chat.completions.create(
            model="gpt-4",
            response_model=Report,
            messages=[{"role": "user", "content": "Write a long report..."}],
            max_retries=3,
        )

    # Should NOT have retried — only one call to the API
    assert call_count == 1


def test_incomplete_output_not_caught_as_retry_exception():
    """Verify IncompleteOutputException is NOT an InstructorRetryException."""
    mock_response = _make_truncated_response()

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    caught_as_retry = False
    caught_as_incomplete = False

    try:
        client.chat.completions.create(
            model="gpt-4",
            response_model=Report,
            messages=[{"role": "user", "content": "Write a long report..."}],
            max_retries=0,
        )
    except InstructorRetryException:
        caught_as_retry = True
    except IncompleteOutputException:
        caught_as_incomplete = True

    assert caught_as_incomplete, "Should be caught as IncompleteOutputException"
    assert not caught_as_retry, "Should NOT be caught as InstructorRetryException"


def test_incomplete_output_exception_tools_mode():
    """IncompleteOutputException should also work in TOOLS mode."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.refusal = None
    mock_response.choices[0].finish_reason = "length"
    mock_response.usage = None

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.TOOLS)

    with pytest.raises(IncompleteOutputException) as exc_info:
        client.chat.completions.create(
            model="gpt-4",
            response_model=Report,
            messages=[{"role": "user", "content": "Write a long report..."}],
            max_retries=0,
        )

    assert exc_info.value.last_completion is mock_response
