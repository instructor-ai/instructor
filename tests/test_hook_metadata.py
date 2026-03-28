"""Tests for retry metadata on completion:error and completion:last_attempt hooks."""

from __future__ import annotations

from instructor.core.hooks import Hooks


def test_completion_error_passes_metadata():
    """Handler accepting **kwargs receives attempt metadata."""
    received = {}

    def handler(_error, **kwargs):
        received.update(kwargs)

    hooks = Hooks()
    hooks.on("completion:error", handler)
    hooks.emit_completion_error(
        RuntimeError("fail"),
        attempt_number=2,
        max_attempts=3,
        is_last_attempt=False,
    )
    assert received["attempt_number"] == 2
    assert received["max_attempts"] == 3
    assert received["is_last_attempt"] is False


def test_completion_last_attempt_passes_metadata():
    """Handler accepting **kwargs receives attempt metadata."""
    received = {}

    def handler(_error, **kwargs):
        received.update(kwargs)

    hooks = Hooks()
    hooks.on("completion:last_attempt", handler)
    hooks.emit_completion_last_attempt(
        RuntimeError("fail"),
        attempt_number=3,
        max_attempts=3,
        is_last_attempt=True,
    )
    assert received["attempt_number"] == 3
    assert received["max_attempts"] == 3
    assert received["is_last_attempt"] is True


def test_backward_compat_old_style_handler():
    """Old-style handler(error) still works when metadata kwargs are emitted."""
    errors = []

    def handler(error):
        errors.append(error)

    hooks = Hooks()
    hooks.on("completion:error", handler)
    err = RuntimeError("fail")
    hooks.emit_completion_error(
        err,
        attempt_number=1,
        max_attempts=3,
        is_last_attempt=False,
    )
    assert len(errors) == 1
    assert errors[0] is err


def test_backward_compat_last_attempt_old_style():
    """Old-style handler(error) works for last_attempt hook too."""
    errors = []

    def handler(error):
        errors.append(error)

    hooks = Hooks()
    hooks.on("completion:last_attempt", handler)
    err = RuntimeError("done")
    hooks.emit_completion_last_attempt(
        err,
        attempt_number=3,
        max_attempts=3,
        is_last_attempt=True,
    )
    assert len(errors) == 1
    assert errors[0] is err


def test_handler_with_specific_kwargs():
    """Handler accepting specific keyword params works."""
    received = {}

    def handler(_error, attempt_number=None, max_attempts=None, is_last_attempt=None):
        received["attempt_number"] = attempt_number
        received["max_attempts"] = max_attempts
        received["is_last_attempt"] = is_last_attempt

    hooks = Hooks()
    hooks.on("completion:error", handler)
    hooks.emit_completion_error(
        RuntimeError("fail"),
        attempt_number=1,
        max_attempts=5,
        is_last_attempt=False,
    )
    assert received["attempt_number"] == 1
    assert received["max_attempts"] == 5
    assert received["is_last_attempt"] is False


def test_no_metadata_still_works():
    """Emitting without metadata still calls handlers normally."""
    errors = []

    def handler(error):
        errors.append(error)

    hooks = Hooks()
    hooks.on("completion:error", handler)
    err = RuntimeError("no meta")
    hooks.emit_completion_error(err)
    assert len(errors) == 1
    assert errors[0] is err
