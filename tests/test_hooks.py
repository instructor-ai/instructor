"""Tests for the Hooks emit system, focusing on attempt metadata passed to
completion:error and completion:last_attempt handlers (issue #2222)."""

import pytest
from instructor.core.hooks import Hooks, HookName


# ---------------------------------------------------------------------------
# completion:error — attempt metadata forwarding
# ---------------------------------------------------------------------------


def test_completion_error_passes_attempt_metadata():
    """completion:error handlers receive attempt_number, max_attempts, is_last_attempt."""
    hooks = Hooks()
    received: dict = {}

    def handler(error: Exception, *, attempt_number: int, max_attempts: int | None, is_last_attempt: bool) -> None:
        received["attempt_number"] = attempt_number
        received["max_attempts"] = max_attempts
        received["is_last_attempt"] = is_last_attempt

    hooks.on(HookName.COMPLETION_ERROR, handler)
    hooks.emit_completion_error(
        ValueError("test"),
        attempt_number=2,
        max_attempts=3,
        is_last_attempt=False,
    )

    assert received["attempt_number"] == 2
    assert received["max_attempts"] == 3
    assert received["is_last_attempt"] is False


def test_completion_error_is_last_attempt_true():
    """is_last_attempt=True is correctly forwarded on the final attempt."""
    hooks = Hooks()
    received: dict = {}

    def handler(error: Exception, *, attempt_number: int, max_attempts: int | None, is_last_attempt: bool) -> None:
        received["is_last_attempt"] = is_last_attempt
        received["attempt_number"] = attempt_number

    hooks.on(HookName.COMPLETION_ERROR, handler)
    hooks.emit_completion_error(
        ValueError("final"),
        attempt_number=3,
        max_attempts=3,
        is_last_attempt=True,
    )

    assert received["is_last_attempt"] is True
    assert received["attempt_number"] == 3


def test_completion_error_max_attempts_none():
    """max_attempts=None (unlimited retries) is forwarded correctly."""
    hooks = Hooks()
    received: dict = {}

    def handler(error: Exception, *, attempt_number: int, max_attempts: int | None, is_last_attempt: bool) -> None:
        received["max_attempts"] = max_attempts

    hooks.on(HookName.COMPLETION_ERROR, handler)
    hooks.emit_completion_error(
        ValueError("test"),
        attempt_number=1,
        max_attempts=None,
        is_last_attempt=False,
    )

    assert received["max_attempts"] is None


def test_completion_error_backward_compatible_handler():
    """Old-style handlers that only accept (error,) still work without errors."""
    hooks = Hooks()
    received: list = []

    def old_style_handler(error: Exception) -> None:
        received.append(error)

    hooks.on(HookName.COMPLETION_ERROR, old_style_handler)
    err = ValueError("test")
    hooks.emit_completion_error(
        err,
        attempt_number=1,
        max_attempts=3,
        is_last_attempt=False,
    )

    assert received == [err]


# ---------------------------------------------------------------------------
# completion:last_attempt — attempt metadata forwarding
# ---------------------------------------------------------------------------


def test_completion_last_attempt_passes_attempt_metadata():
    """completion:last_attempt handlers receive attempt_number, max_attempts, is_last_attempt."""
    hooks = Hooks()
    received: dict = {}

    def handler(error: Exception, *, attempt_number: int, max_attempts: int | None, is_last_attempt: bool) -> None:
        received["attempt_number"] = attempt_number
        received["max_attempts"] = max_attempts
        received["is_last_attempt"] = is_last_attempt

    hooks.on(HookName.COMPLETION_LAST_ATTEMPT, handler)
    hooks.emit_completion_last_attempt(
        ValueError("final error"),
        attempt_number=3,
        max_attempts=3,
        is_last_attempt=True,
    )

    assert received["attempt_number"] == 3
    assert received["max_attempts"] == 3
    assert received["is_last_attempt"] is True


def test_completion_last_attempt_backward_compatible_handler():
    """Old-style handlers that only accept (error,) still work without errors."""
    hooks = Hooks()
    received: list = []

    def old_style_handler(error: Exception) -> None:
        received.append(error)

    hooks.on(HookName.COMPLETION_LAST_ATTEMPT, old_style_handler)
    err = RuntimeError("final")
    hooks.emit_completion_last_attempt(
        err,
        attempt_number=2,
        max_attempts=2,
        is_last_attempt=True,
    )

    assert received == [err]


# ---------------------------------------------------------------------------
# Multiple handlers — all receive metadata
# ---------------------------------------------------------------------------


def test_multiple_handlers_all_receive_metadata():
    """When multiple handlers are registered, all receive attempt metadata."""
    hooks = Hooks()
    calls: list[dict] = []

    def handler_a(error: Exception, *, attempt_number: int, max_attempts: int | None, is_last_attempt: bool) -> None:
        calls.append({"handler": "a", "attempt_number": attempt_number})

    def handler_b(error: Exception, *, attempt_number: int, max_attempts: int | None, is_last_attempt: bool) -> None:
        calls.append({"handler": "b", "attempt_number": attempt_number})

    hooks.on(HookName.COMPLETION_ERROR, handler_a)
    hooks.on(HookName.COMPLETION_ERROR, handler_b)
    hooks.emit_completion_error(
        ValueError("test"),
        attempt_number=2,
        max_attempts=5,
        is_last_attempt=False,
    )

    assert len(calls) == 2
    assert all(c["attempt_number"] == 2 for c in calls)


def test_mixed_handlers_old_and_new_style():
    """Old-style and new-style handlers can coexist on the same hook."""
    hooks = Hooks()
    old_calls: list = []
    new_calls: list[dict] = []

    def old_handler(error: Exception) -> None:
        old_calls.append(error)

    def new_handler(error: Exception, *, attempt_number: int, max_attempts: int | None, is_last_attempt: bool) -> None:
        new_calls.append({"attempt_number": attempt_number, "is_last_attempt": is_last_attempt})

    hooks.on(HookName.COMPLETION_ERROR, old_handler)
    hooks.on(HookName.COMPLETION_ERROR, new_handler)

    err = ValueError("test")
    hooks.emit_completion_error(
        err,
        attempt_number=1,
        max_attempts=3,
        is_last_attempt=False,
    )

    assert old_calls == [err]
    assert new_calls == [{"attempt_number": 1, "is_last_attempt": False}]
