import logging

from instructor.mode import Mode
from instructor.processing.response import handle_response_model


def test_handle_response_model_redacts_api_key_in_logs(caplog):
    caplog.set_level(logging.DEBUG, logger="instructor")

    handle_response_model(
        response_model=None,
        mode=Mode.TOOLS,
        api_key="sk-test-123456",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "sk-test-123456" not in logs
    assert "***REDACTED***" in logs


def test_handle_response_model_redacts_sensitive_values_in_extra(caplog):
    caplog.set_level(logging.DEBUG, logger="instructor")

    handle_response_model(
        response_model=None,
        mode=Mode.TOOLS,
        api_key="sk-secret-extra",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )

    matching = [
        record
        for record in caplog.records
        if "Instructor Request:" in record.getMessage()
    ]
    assert matching

    record = matching[-1]
    assert record.__dict__["new_kwargs"]["api_key"] == "***REDACTED***"