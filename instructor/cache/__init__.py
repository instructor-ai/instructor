"""Caching utilities for Instructor.

This module provides a very small abstraction layer so that users can
plug different cache back-ends (in-process LRU, `diskcache`, `redis`, …)
into the Instructor client via the ``cache=...`` keyword::

    from instructor import from_provider
    from instructor.cache import AutoCache

    cache = AutoCache(maxsize=10_000)
    client = from_provider("openai/gpt-4o", cache=cache)

The cache object must implement :class:`BaseCache`.  A minimal
requirement is to expose synchronous ``get`` / ``set`` methods (async
wrappers currently call them directly).  The default implementation
``AutoCache`` is an in-process LRU cache with a configurable size.

This first iteration purposefully keeps the API narrow: no eviction
hooks, no invalidation, no TTL for the LRU variant.  The objective is to
provide a safe foundation which we will extend in follow-up work.
"""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any
import logging

# The project already depends on pydantic; type checker in some
# environments might not have its stubs – silence if missing.
from pydantic import BaseModel  # type: ignore[import-not-found]

__all__ = [
    "BaseCache",
    "AutoCache",
    "DiskCache",
    "make_cache_key",
]


class BaseCache(ABC):
    """Abstract cache contract.

    Concrete subclasses *must* be thread-safe.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:  # noqa: ANN401 – value type arbitrary
        """Return *None* to indicate a cache miss."""

    @abstractmethod
    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,  # noqa: ARG002
    ) -> None:  # noqa: ANN401
        """Store *value* under *key*.

        ``ttl`` is time-to-live in **seconds**.  Implementations *may*
        ignore it (e.g. :class:`AutoCache`).
        """


class AutoCache(BaseCache):
    """Thread-safe in-process LRU cache using :class:`collections.OrderedDict`."""

    def __init__(self, maxsize: int = 128):
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------------
    # BaseCache implementation
    # ---------------------------------------------------------------------
    def get(self, key: str) -> Any | None:  # noqa: ANN401
        with self._lock:
            try:
                value = self._cache.pop(key)
            except KeyError:
                return None
            # Move to the end (most recently used)
            self._cache[key] = value
            return value

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,  # noqa: ARG002
    ) -> None:  # noqa: ANN401
        # *ttl* is ignored for the in-process cache.
        with self._lock:
            if key in self._cache:
                self._cache.pop(key, None)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                # popitem(last=False) pops the *least* recently used entry
                self._cache.popitem(last=False)


# -------------------------------------------------------------------------
# Optional back-ends – imported lazily so users do not need extra deps
# -------------------------------------------------------------------------


def _import_diskcache():  # pragma: no cover – only executed when requested
    import importlib  # type: ignore[]

    if importlib.util.find_spec("diskcache") is None:  # type: ignore[attr-defined]
        raise ImportError(
            "diskcache is not installed.  Install it with `pip install diskcache`."
        )
    import diskcache  # type: ignore

    return diskcache


class DiskCache(BaseCache):
    """Wrapper around `diskcache.Cache`."""

    def __init__(self, directory: str = ".instructor_cache", **kwargs: Any):
        diskcache = _import_diskcache()
        self._cache = diskcache.Cache(directory, **kwargs)

    def get(self, key: str) -> Any | None:  # noqa: ANN401
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:  # noqa: ANN401
        if ttl is None:
            self._cache.set(key, value)
        else:
            self._cache.set(key, value, expire=ttl)


# -------------------------------------------------------------------------
# Cache-key helper
# -------------------------------------------------------------------------


def make_cache_key(
    *,
    messages: Any,
    model: str | None,
    response_model: type[BaseModel] | None,
    mode: str | None = None,
) -> str:  # noqa: ANN401
    """Compute a *deterministic* cache key.

    The key space uses SHA-256("json payload") to keep the final length
    fixed regardless of input size.

    Components that influence the key:
        • provider/model name
        • serialized *messages* (user + system prompt, etc.)
        • *mode* (Tools, JSON, …) – helps when users change Instructor mode
        • *response_model* schema – so edits to field definitions or
          descriptions invalidate prior cache entries (critical!).
    """

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "mode": mode,
    }

    if response_model is not None:
        # Include the entire JSON schema – guarantees busting when either
        # a field or its meta (title, description, constraints) changes.
        payload["schema"] = response_model.model_json_schema()

    # ``default=str`` converts non-serializable objects (e.g. datetime) to
    # string so dumps never fails.
    data = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()


# -------------------------------------------------------------------------
# Convenience helpers used by patch.py to avoid duplication
# -------------------------------------------------------------------------

logger = logging.getLogger("instructor.cache")


def load_cached_response(cache: BaseCache, key: str, response_model: type[BaseModel]):  # noqa: ANN201
    """Return parsed model if *key* exists in *cache* else None."""
    cached = cache.get(key)
    if cached is None:
        return None
    import json

    try:
        data = json.loads(cached)
        model_json = data["model"]
        raw_json = data.get("raw")
    except Exception:  # noqa: BLE001
        model_json = cached
        raw_json = None

    obj = response_model.model_validate_json(model_json)  # type: ignore[arg-type]
    if raw_json is not None:
        # `_raw_response` is an internal attribute used by Instructor; it may not
        # be declared on the Pydantic model type.
        try:
            # Try to deserialize as JSON and reconstruct object structure
            import json

            raw_data = json.loads(raw_json)

            # Check if this looks like a Pydantic-serialized object (has proper structure)
            if isinstance(raw_data, dict) and any(
                key in raw_data for key in ["id", "object", "model", "choices"]
            ):
                # Looks like a proper completion object - use SimpleNamespace reconstruction
                from types import SimpleNamespace

                obj._raw_response = json.loads(
                    raw_json, object_hook=lambda d: SimpleNamespace(**d)
                )  # type: ignore[attr-defined]
                logger.debug("Restored raw response as SimpleNamespace object")
            else:
                # Plain dict/list - keep as-is
                obj._raw_response = raw_data  # type: ignore[attr-defined]
                logger.debug("Restored raw response as plain data structure")
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON - probably string fallback
            obj._raw_response = raw_json  # type: ignore[attr-defined]
            logger.debug(
                "Restored raw response as string (original could not be fully serialized)"
            )
    logger.debug("cache hit: %s", key)
    return obj


def store_cached_response(
    cache: BaseCache, key: str, model: BaseModel, ttl: int | None = None
) -> None:  # noqa: D401
    """Serialize *model* and optional raw response to JSON and cache it."""
    import json

    raw_resp = getattr(model, "_raw_response", None)
    if raw_resp is not None:
        try:
            # Try Pydantic model serialization first (OpenAI, Anthropic, etc.)
            raw_json = raw_resp.model_dump_json()  # type: ignore[attr-defined]
            logger.debug("Cached raw response as Pydantic JSON")
        except (AttributeError, TypeError) as e:
            # Fallback for non-Pydantic responses (custom providers, plain dicts, etc.)
            try:
                import json

                raw_json = json.dumps(raw_resp, default=str)
                logger.debug(
                    "Cached raw response as plain JSON (provider may not support full reconstruction)"
                )
            except (TypeError, ValueError):
                # Final fallback - string representation
                raw_json = str(raw_resp)
                logger.warning(
                    "Raw response could not be serialized as JSON, using string fallback. "
                    "create_with_completion may not fully restore original object structure."
                )
    else:
        raw_json = None

    payload = {
        "model": model.model_dump_json(),  # type: ignore[attr-defined]
        "raw": raw_json,
    }
    cache.set(key, json.dumps(payload), ttl=ttl)
    logger.debug("cache store: %s", key)
