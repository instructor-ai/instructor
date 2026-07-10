"""Tests for ModeRegistry thread safety (issue #2422).

Verifies that concurrent first-access to a lazy-loaded mode handler does not
cause permanent KeyError/RegistryError for losing threads.
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import ModeRegistry, ModeHandlers


def _make_dummy_handlers() -> ModeHandlers:
    """Create minimal ModeHandlers for testing."""
    def dummy_request(response_model=None, kwargs=None):
        return kwargs or {}

    def dummy_reask(kwargs=None, response=None, exception=None):
        return kwargs or {}

    def dummy_response(response, response_model=None, validation_context=None, mode=None, stream=None, **kw):
        return response

    return ModeHandlers(
        request_handler=dummy_request,
        reask_handler=dummy_reask,
        response_parser=dummy_response,
    )


class TestRegistryThreadSafety:
    """Tests that ModeRegistry.get_handlers() is thread-safe."""

    def test_concurrent_first_access_no_failures(self):
        """All threads should succeed when concurrently resolving the same
        lazy-loaded mode for the first time.

        Before the fix, 49/50 threads would fail with KeyError because the
        first thread pops the lazy loader entry before others can use it.
        """
        registry = ModeRegistry()
        call_count = 0
        count_lock = threading.Lock()

        def slow_loader():
            """Simulate a slow module import to widen the race window."""
            nonlocal call_count
            with count_lock:
                call_count += 1
            # Small delay to ensure threads overlap
            threading.Event().wait(0.01)
            return _make_dummy_handlers()

        registry.register_lazy(
            Provider.OPENAI, Mode.TOOLS, slow_loader
        )

        results: list[tuple[int, str]] = []
        results_lock = threading.Lock()

        def check(i: int) -> tuple[int, str]:
            try:
                handlers = registry.get_handlers(Provider.OPENAI, Mode.TOOLS)
                return (i, "OK")
            except Exception as e:
                return (i, f"{type(e).__name__}: {str(e)[:80]}")

        with ThreadPoolExecutor(max_workers=10) as pool:
            raw_results = list(pool.map(check, range(50)))

        with results_lock:
            results = raw_results

        failures = [r for r in results if r[1] != "OK"]
        assert not failures, (
            f"{len(failures)}/50 threads failed under concurrent access:\n"
            + "\n".join(f"  thread {i}: {err}" for i, err in failures[:5])
        )

    def test_loader_called_exactly_once(self):
        """The lazy loader should be called exactly once even under heavy
        concurrency, because the lock serializes the lazy-load path."""
        registry = ModeRegistry()
        call_count = 0
        count_lock = threading.Lock()

        def loader():
            nonlocal call_count
            with count_lock:
                call_count += 1
            threading.Event().wait(0.01)
            return _make_dummy_handlers()

        registry.register_lazy(Provider.OPENAI, Mode.TOOLS, loader)

        with ThreadPoolExecutor(max_workers=10) as pool:
            list(pool.map(lambda _: registry.get_handlers(Provider.OPENAI, Mode.TOOLS), range(30)))

        assert call_count == 1, (
            f"Loader was called {call_count} times, expected exactly 1. "
            "The lock should serialize lazy loading."
        )

    def test_double_checked_locking_resolves_quickly(self):
        """After the first thread resolves the handler, subsequent threads
        should get it from _handlers (fast path), not re-enter the lock."""
        registry = ModeRegistry()

        def loader():
            return _make_dummy_handlers()

        registry.register_lazy(Provider.OPENAI, Mode.TOOLS, loader)

        # First call triggers lazy load
        registry.get_handlers(Provider.OPENAI, Mode.TOOLS)
        # The loader entry should be consumed
        assert (Provider.OPENAI, Mode.TOOLS) not in registry._lazy_loaders
        # The handler should be cached
        assert (Provider.OPENAI, Mode.TOOLS) in registry._handlers

        # Subsequent calls should hit the fast path
        handlers2 = registry.get_handlers(Provider.OPENAI, Mode.TOOLS)
        assert handlers2 is registry._handlers[(Provider.OPENAI, Mode.TOOLS)]

    def test_unregistered_mode_still_raises(self):
        """An unregistered mode should still raise KeyError, not silently
        succeed or hang."""
        registry = ModeRegistry()

        with pytest.raises(KeyError, match="is not registered"):
            registry.get_handlers(Provider.OPENAI, Mode.TOOLS)

    def test_concurrent_different_modes(self):
        """Concurrent access to different mode keys should not interfere."""
        registry = ModeRegistry()

        def make_loader(p, m):
            return lambda: _make_dummy_handlers()

        modes = [
            (Provider.OPENAI, Mode.TOOLS),
            (Provider.ANTHROPIC, Mode.TOOLS),
            (Provider.GEMINI, Mode.TOOLS),
        ]

        for provider, mode in modes:
            registry.register_lazy(provider, mode, make_loader(provider, mode))

        def check(key):
            provider, mode = key
            try:
                registry.get_handlers(provider, mode)
                return "OK"
            except Exception as e:
                return f"{type(e).__name__}: {e}"

        # Each key accessed by multiple threads simultaneously
        keys = modes * 10  # 30 tasks, 10 per key
        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(check, keys))

        failures = [r for r in results if r != "OK"]
        assert not failures, f"Concurrent different-mode access failed: {failures}"
