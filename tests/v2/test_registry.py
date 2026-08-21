"""Tests for v2 mode registry."""

from typing import Any, cast

import pytest

from instructor import Mode
from instructor.v2 import Provider, mode_registry
from instructor.v2.core.decorators import register_mode_handler
from tests.v2.conftest import get_registered_provider_mode_pairs


def _get_registered_providers() -> list[Provider]:
    pairs = get_registered_provider_mode_pairs()
    return sorted({provider for provider, _ in pairs}, key=lambda p: p.value)


def _get_registered_modes() -> list[Mode]:
    pairs = get_registered_provider_mode_pairs()
    return sorted({mode for _, mode in pairs}, key=lambda m: m.value)


def _get_registered_provider_modes() -> list[tuple[Provider, Mode]]:
    return get_registered_provider_mode_pairs()


def test_registry_registration():
    """Test basic registration."""

    @register_mode_handler(Provider.DEEPSEEK, Mode.JSON)
    class TestHandler:
        def prepare_request(self, response_model, kwargs):
            return response_model, kwargs

        def handle_reask(self, kwargs, _response, _exception):
            return kwargs

        def parse_response(self, _response, response_model, **_kwargs):
            return response_model()

    # Check it's registered
    assert mode_registry.is_registered(Provider.DEEPSEEK, Mode.JSON)

    # Get handlers
    handlers = mode_registry.get_handlers(Provider.DEEPSEEK, Mode.JSON)
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


def test_registry_get_handler():
    """Test getting specific handler types."""

    @register_mode_handler(Provider.OPENROUTER, Mode.TOOLS)
    class TestHandler:
        def prepare_request(self, response_model, _kwargs):
            return response_model, {"test": "request"}

        def handle_reask(self, _kwargs, _response, _exception):
            return {"test": "reask"}

        def parse_response(self, _response, response_model, **_kwargs):
            return response_model()

    # Get individual handlers
    request_handler = mode_registry.get_handler(
        Provider.OPENROUTER, Mode.TOOLS, "request"
    )
    result = request_handler(None, {})
    assert result[1]["test"] == "request"

    reask_handler = mode_registry.get_handler(Provider.OPENROUTER, Mode.TOOLS, "reask")
    result = reask_handler({}, None, None)
    assert result["test"] == "reask"


@pytest.mark.parametrize("provider", _get_registered_providers())
def test_registry_query_by_provider(provider: Provider):
    """Test querying modes for a provider."""
    modes = mode_registry.get_modes_for_provider(provider)
    assert modes, f"{provider.value} should have at least one mode"

    expected_modes = {
        mode for prov, mode in get_registered_provider_mode_pairs() if prov == provider
    }
    assert expected_modes.issubset(set(modes))


@pytest.mark.parametrize("mode", _get_registered_modes())
def test_registry_query_by_mode_type(mode: Mode):
    """Test querying providers for a mode type."""
    providers = mode_registry.get_providers_for_mode(mode)
    assert providers, f"{mode.value} should have at least one provider"

    expected_providers = {
        provider
        for provider, registered_mode in get_registered_provider_mode_pairs()
        if registered_mode == mode
    }
    assert expected_providers.issubset(set(providers))


@pytest.mark.parametrize("provider,mode", _get_registered_provider_modes())
def test_registry_list_modes(provider: Provider, mode: Mode):
    """Test listing all registered modes."""
    all_modes = mode_registry.list_modes()
    assert (provider, mode) in all_modes


def test_registry_not_registered():
    """Test error when mode not registered."""
    with pytest.raises(KeyError, match="not registered"):
        mode_registry.get_handlers(Provider.GEMINI, Mode.JSON_SCHEMA)


@pytest.mark.parametrize("provider,mode", _get_registered_provider_modes())
def test_registry_invalid_handler_type(provider: Provider, mode: Mode):
    """Test error for invalid handler type."""
    with pytest.raises(ValueError, match="Invalid handler_type"):
        mode_registry.get_handler(provider, mode, "invalid_type")


def test_get_handlers_concurrent_first_access_does_not_race():
    """Regression test for #2422.

    Concurrent first callers for the same lazily-registered (provider, mode)
    key must all get the same handlers back, never a KeyError, even when the
    loader is slow (simulating a module import in flight).

    The loader sleeps briefly so every other thread has a chance to reach
    get_handlers() while the first thread's load is still in progress, this
    is the exact window in which the unlocked version raced: losing threads
    would see the lazy loader already popped and self._handlers not yet set,
    and raise KeyError for a mode that genuinely is registered.
    """
    import threading
    import time

    from instructor.v2.core.registry import ModeHandlers, ModeRegistry

    registry = ModeRegistry()
    n_threads = 8
    started = threading.Barrier(n_threads, timeout=5)
    load_count = 0
    load_count_lock = threading.Lock()

    def slow_loader() -> ModeHandlers:
        nonlocal load_count
        with load_count_lock:
            load_count += 1
        # Simulate a slow module import: gives every other thread time to
        # reach get_handlers() and start contending before this resolves.
        time.sleep(0.2)
        return ModeHandlers(
            request_handler=cast(Any, lambda *_a, **_k: None),
            reask_handler=cast(Any, lambda *_a, **_k: None),
            response_parser=cast(Any, lambda *_a, **_k: None),
        )

    registry.register_lazy(Provider.DEEPSEEK, Mode.TOOLS, slow_loader)

    results: list[object] = [None] * n_threads
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(idx: int) -> None:
        # Line up all threads so they call get_handlers() as close to
        # simultaneously as possible, maximizing contention.
        started.wait(timeout=5)
        try:
            results[idx] = registry.get_handlers(Provider.DEEPSEEK, Mode.TOOLS)
        except BaseException as exc:  # noqa: BLE001
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == [], f"get_handlers raised for concurrent callers: {errors}"
    assert all(r is not None for r in results)
    assert len({id(r) for r in results}) == 1, (
        "all callers should get the same ModeHandlers instance"
    )
    # The loader must run exactly once, not once per racing thread.
    assert load_count == 1


def test_is_registered_and_list_modes_during_concurrent_lazy_load():
    """Regression test for #2535.

    When thread A is in the middle of resolving a lazy loader, concurrent
    readers calling is_registered() or list_modes() must still see the mode
    as registered rather than raising RegistryError due to a momentary gap
    between popping _lazy_loaders and publishing _handlers.
    """
    import threading
    import time
    from typing import cast

    from instructor.v2.core.registry import ModeHandlers, ModeRegistry

    registry = ModeRegistry()
    load_in_progress = threading.Event()
    finish_load = threading.Event()

    def slow_loader() -> ModeHandlers:
        load_in_progress.set()
        finish_load.wait(timeout=5)
        return ModeHandlers(
            request_handler=cast(Any, lambda *_a, **_k: None),
            reask_handler=cast(Any, lambda *_a, **_k: None),
            response_parser=cast(Any, lambda *_a, **_k: None),
        )

    registry.register_lazy(Provider.OPENAI, Mode.JSON, slow_loader)

    # Thread 1 starts lazy load
    t1 = threading.Thread(target=lambda: registry.get_handlers(Provider.OPENAI, Mode.JSON))
    t1.start()

    # Wait until thread 1 has popped the loader and is executing the slow loader
    assert load_in_progress.wait(timeout=5)

    # Thread 2 checks registration while thread 1 is still inside the loader
    assert registry.is_registered(Provider.OPENAI, Mode.JSON), (
        "is_registered() must return True even while lazy loading is in-flight"
    )
    assert (Provider.OPENAI, Mode.JSON) in registry.list_modes(), (
        "list_modes() must include mode even while lazy loading is in-flight"
    )

    finish_load.set()
    t1.join(timeout=5)

    # Post-load sanity check
    assert registry.is_registered(Provider.OPENAI, Mode.JSON)
    assert (Provider.OPENAI, Mode.JSON) in registry.list_modes()

