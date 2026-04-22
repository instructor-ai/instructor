"""Tests for retry metadata exposed via completion:error and completion:last_attempt hooks.

Verifies that handlers registered on these hooks receive attempt_number,
max_attempts, and is_last_attempt as keyword arguments (issue #2222).
"""

from instructor.core.hooks import Hooks, HookName


class TestCompletionErrorMetadata:
    def test_attempt_number_forwarded(self):
        hooks = Hooks()
        received: dict = {}

        def handler(error: Exception, *, attempt_number: int = 0, **kw):
            received["attempt_number"] = attempt_number

        hooks.on(HookName.COMPLETION_ERROR, handler)
        hooks.emit_completion_error(ValueError("boom"), attempt_number=3, max_attempts=5, is_last_attempt=False)

        assert received["attempt_number"] == 3

    def test_max_attempts_forwarded(self):
        hooks = Hooks()
        received: dict = {}

        def handler(error: Exception, *, max_attempts=None, **kw):
            received["max_attempts"] = max_attempts

        hooks.on(HookName.COMPLETION_ERROR, handler)
        hooks.emit_completion_error(ValueError("boom"), attempt_number=1, max_attempts=5, is_last_attempt=False)

        assert received["max_attempts"] == 5

    def test_is_last_attempt_false(self):
        hooks = Hooks()
        received: dict = {}

        def handler(error: Exception, *, is_last_attempt: bool = True, **kw):
            received["is_last_attempt"] = is_last_attempt

        hooks.on(HookName.COMPLETION_ERROR, handler)
        hooks.emit_completion_error(ValueError("boom"), attempt_number=1, max_attempts=3, is_last_attempt=False)

        assert received["is_last_attempt"] is False

    def test_is_last_attempt_true(self):
        hooks = Hooks()
        received: dict = {}

        def handler(error: Exception, *, is_last_attempt: bool = False, **kw):
            received["is_last_attempt"] = is_last_attempt

        hooks.on(HookName.COMPLETION_ERROR, handler)
        hooks.emit_completion_error(ValueError("boom"), attempt_number=3, max_attempts=3, is_last_attempt=True)

        assert received["is_last_attempt"] is True

    def test_handler_without_metadata_params_still_called(self):
        """Handlers that only accept error (old-style) must still work."""
        hooks = Hooks()
        received: list = []

        def handler(error: Exception):
            received.append(error)

        hooks.on(HookName.COMPLETION_ERROR, handler)
        err = ValueError("boom")
        hooks.emit_completion_error(err, attempt_number=1, max_attempts=3, is_last_attempt=False)

        assert received == [err]

    def test_max_attempts_none_when_unbounded(self):
        hooks = Hooks()
        received: dict = {}

        def handler(error: Exception, *, max_attempts=..., **kw):
            received["max_attempts"] = max_attempts

        hooks.on(HookName.COMPLETION_ERROR, handler)
        hooks.emit_completion_error(ValueError("boom"), attempt_number=1, max_attempts=None, is_last_attempt=False)

        assert received["max_attempts"] is None


class TestCompletionLastAttemptMetadata:
    def test_attempt_number_forwarded(self):
        hooks = Hooks()
        received: dict = {}

        def handler(error: Exception, *, attempt_number: int = 0, **kw):
            received["attempt_number"] = attempt_number

        hooks.on(HookName.COMPLETION_LAST_ATTEMPT, handler)
        hooks.emit_completion_last_attempt(ValueError("final"), attempt_number=5, max_attempts=5, is_last_attempt=True)

        assert received["attempt_number"] == 5

    def test_is_last_attempt_always_true(self):
        hooks = Hooks()
        received: dict = {}

        def handler(error: Exception, *, is_last_attempt: bool = False, **kw):
            received["is_last_attempt"] = is_last_attempt

        hooks.on(HookName.COMPLETION_LAST_ATTEMPT, handler)
        hooks.emit_completion_last_attempt(ValueError("final"), attempt_number=3, max_attempts=3, is_last_attempt=True)

        assert received["is_last_attempt"] is True

    def test_handler_without_metadata_params_still_called(self):
        """Old-style handlers that only accept error must still work."""
        hooks = Hooks()
        received: list = []

        def handler(error: Exception):
            received.append(error)

        hooks.on(HookName.COMPLETION_LAST_ATTEMPT, handler)
        err = ValueError("final")
        hooks.emit_completion_last_attempt(err, attempt_number=3, max_attempts=3, is_last_attempt=True)

        assert received == [err]
