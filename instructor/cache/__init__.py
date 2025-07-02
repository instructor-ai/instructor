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

# The project already depends on pydantic; type checker in some
# environments might not have its stubs – silence if missing.
from pydantic import BaseModel  # type: ignore[import-not-found]

__all__ = [
    "BaseCache",
    "AutoCache",
    "DiskCache",
    "RedisCache",
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
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:  # noqa: ANN401
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

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:  # noqa: ANN401
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

    if importlib.util.find_spec("diskcache") is None:
        raise ImportError(
            "diskcache is not installed.  Install it with `pip install diskcache`."
        )
    import diskcache  # type: ignore

    return diskcache  # noqa: WPS331 – re-export helper


def _import_redis():  # pragma: no cover – only executed when requested
    import importlib  # type: ignore[]

    if importlib.util.find_spec("redis") is None:
        raise ImportError("redis is not installed.  Install it with `pip install redis`.")
    import redis  # type: ignore

    return redis


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


class RedisCache(BaseCache):
    """Thin wrapper around `redis.Redis`.  Works with sync API for now."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, **kwargs: Any):
        redis = _import_redis()
        self._r = redis.Redis(host=host, port=port, db=db, **kwargs)

    def get(self, key: str) -> Any | None:  # noqa: ANN401
        value = self._r.get(key)
        return value  # type: ignore[return-value]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:  # noqa: ANN401
        if ttl is None:
            self._r.set(key, value)
        else:
            self._r.setex(key, ttl, value)


# -------------------------------------------------------------------------
# Cache-key helper
# -------------------------------------------------------------------------

def make_cache_key(*, messages: Any, model: str | None, response_model: type[BaseModel] | None, mode: str | None = None) -> str:  # noqa: ANN401
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