"""Tests for cache observability hooks (CACHE_HIT, CACHE_MISS).

These tests verify that cache hooks are properly emitted when
using the caching functionality.
"""

import pytest
from pydantic import BaseModel

from instructor.cache import AutoCache, load_cached_response, store_cached_response
from instructor.core.hooks import Hooks, HookName


class User(BaseModel):
    name: str
    age: int


class TestCacheHooksEnum:
    """Test that cache hooks are properly defined in the enum."""

    def test_cache_hit_in_enum(self):
        assert HookName.CACHE_HIT.value == "cache:hit"

    def test_cache_miss_in_enum(self):
        assert HookName.CACHE_MISS.value == "cache:miss"

    def test_cache_hooks_can_be_retrieved_by_string(self):
        hooks = Hooks()
        assert hooks.get_hook_name("cache:hit") == HookName.CACHE_HIT
        assert hooks.get_hook_name("cache:miss") == HookName.CACHE_MISS


class TestCacheHooksEmit:
    """Test that cache hooks emit correctly."""

    def test_emit_cache_hit(self):
        hooks = Hooks()
        received = []

        def handler(key: str, response):
            received.append({"key": key, "response": response})

        hooks.on(HookName.CACHE_HIT, handler)
        hooks.emit_cache_hit(key="test-key", response={"data": "test"})

        assert len(received) == 1
        assert received[0]["key"] == "test-key"
        assert received[0]["response"] == {"data": "test"}

    def test_emit_cache_miss(self):
        hooks = Hooks()
        received = []

        def handler(key: str, **kwargs):  # noqa: ARG001
            received.append({"key": key})

        hooks.on(HookName.CACHE_MISS, handler)
        hooks.emit_cache_miss(key="test-key")

        assert len(received) == 1
        assert received[0]["key"] == "test-key"


class TestLoadCachedResponseHooks:
    """Test that load_cached_response emits appropriate hooks."""

    def test_emits_cache_miss_when_key_not_found(self):
        cache = AutoCache()
        hooks = Hooks()
        missed_keys = []

        def on_miss(key: str, **kwargs):  # noqa: ARG001
            missed_keys.append(key)

        hooks.on(HookName.CACHE_MISS, on_miss)

        result = load_cached_response(cache, "nonexistent-key", User, hooks=hooks)

        assert result is None
        assert len(missed_keys) == 1
        assert missed_keys[0] == "nonexistent-key"

    def test_emits_cache_hit_when_key_found(self):
        cache = AutoCache()
        hooks = Hooks()
        hit_data = []

        def on_hit(key: str, response):
            hit_data.append({"key": key, "response": response})

        hooks.on(HookName.CACHE_HIT, on_hit)

        # Store a user in cache first
        user = User(name="Alice", age=30)
        store_cached_response(cache, "user-key", user)

        # Now load it
        result = load_cached_response(cache, "user-key", User, hooks=hooks)

        assert result is not None
        assert result.name == "Alice"
        assert result.age == 30
        assert len(hit_data) == 1
        assert hit_data[0]["key"] == "user-key"
        assert hit_data[0]["response"].name == "Alice"

    def test_no_hooks_emitted_when_hooks_is_none(self):
        cache = AutoCache()

        # Should not raise even without hooks
        result = load_cached_response(cache, "nonexistent-key", User, hooks=None)
        assert result is None

        # Store and load without hooks
        user = User(name="Bob", age=25)
        store_cached_response(cache, "bob-key", user)
        result = load_cached_response(cache, "bob-key", User, hooks=None)
        assert result is not None
        assert result.name == "Bob"


class TestCacheHooksIntegration:
    """Integration tests for cache hooks with multiple handlers."""

    def test_multiple_handlers_receive_events(self):
        hooks = Hooks()
        handler1_calls = []
        handler2_calls = []

        def handler1(key: str, **kwargs):  # noqa: ARG001
            handler1_calls.append(key)

        def handler2(key: str, **kwargs):  # noqa: ARG001
            handler2_calls.append(key)

        hooks.on(HookName.CACHE_MISS, handler1)
        hooks.on(HookName.CACHE_MISS, handler2)

        cache = AutoCache()
        load_cached_response(cache, "test-key", User, hooks=hooks)

        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

    def test_can_track_hit_ratio(self):
        """Example of using hooks to track cache hit ratio."""
        cache = AutoCache()
        hooks = Hooks()
        stats = {"hits": 0, "misses": 0}

        def on_hit(**kwargs):  # noqa: ARG001
            stats["hits"] += 1

        def on_miss(**kwargs):  # noqa: ARG001
            stats["misses"] += 1

        hooks.on(HookName.CACHE_HIT, on_hit)
        hooks.on(HookName.CACHE_MISS, on_miss)

        # First call - miss
        load_cached_response(cache, "key1", User, hooks=hooks)

        # Store something
        user = User(name="Test", age=20)
        store_cached_response(cache, "key1", user)

        # Second call - hit
        load_cached_response(cache, "key1", User, hooks=hooks)

        # Third call - miss (different key)
        load_cached_response(cache, "key2", User, hooks=hooks)

        assert stats["hits"] == 1
        assert stats["misses"] == 2

        hit_ratio = stats["hits"] / (stats["hits"] + stats["misses"])
        assert hit_ratio == pytest.approx(1 / 3)
