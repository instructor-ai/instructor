"""Tests for sensitive-key redaction in debug logging.

Regression tests for https://github.com/jxnl/instructor/issues/2265
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from instructor.mode import Mode
from instructor.processing.response import _redact_kwargs, handle_response_model


class User(BaseModel):
    name: str
    age: int


# -- unit tests for _redact_kwargs ------------------------------------------


def test_redact_kwargs_scrubs_sensitive_keys():
    kwargs = {
        "model": "gpt-4",
        "api_key": "sk-secret-123",
        "api_secret": "secret-456",
        "api_token": "tok-789",
        "authorization": "Bearer xxx",
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = _redact_kwargs(kwargs)
    assert result["api_key"] == "[REDACTED]"
    assert result["api_secret"] == "[REDACTED]"
    assert result["api_token"] == "[REDACTED]"
    assert result["authorization"] == "[REDACTED]"


def test_redact_kwargs_preserves_safe_keys():
    kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    result = _redact_kwargs(kwargs)
    assert result == kwargs


def test_redact_kwargs_does_not_mutate_original():
    kwargs = {"api_key": "sk-real-key", "model": "gpt-4"}
    _redact_kwargs(kwargs)
    assert kwargs["api_key"] == "sk-real-key"


def test_redact_kwargs_empty_dict():
    assert _redact_kwargs({}) == {}


# -- integration test: handle_response_model does not leak api_key -----------


def test_handle_response_model_redacts_api_key_in_debug_log(caplog):
    """Ensure api_key is never written to log output (issue #2265)."""
    secret = "sk-SUPER-SECRET-KEY-12345"

    with caplog.at_level(logging.DEBUG, logger="instructor"):
        _, new_kwargs = handle_response_model(
            response_model=User,
            mode=Mode.TOOLS,
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            api_key=secret,
        )

    # The returned kwargs must still carry the real key for the API call.
    assert new_kwargs["api_key"] == secret

    # But the log output must never contain the real key.
    for record in caplog.records:
        assert secret not in record.getMessage(), (
            f"api_key leaked in log record: {record.getMessage()}"
        )
        # Also check the extra dict attached to the record
        if hasattr(record, "new_kwargs"):
            assert secret not in str(record.new_kwargs), (
                "api_key leaked in log record extra"
            )
