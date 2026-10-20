"""Serialize and restore cached runtime responses."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from instructor.v2.validation.async_validators import reject_async_validators

if TYPE_CHECKING:
    from instructor.cache import BaseCache

logger = logging.getLogger("instructor.cache")

# Keys written by store_cached_response. Older envelopes may omit "raw",
# so recognition must accept the subset, not require both keys.
_ENVELOPE_KEYS = frozenset({"model", "raw"})


def _is_response_envelope(data: Any) -> bool:
    """True when *data* looks like the envelope written by store_cached_response.

    A legacy cache entry stores the response model's own JSON directly, which
    may itself contain a business field named "model" (or even "raw"). The
    envelope is only recognized when every key belongs to the envelope and
    "model" holds a JSON document string, as the writer always stores the
    serialized model as a JSON string.
    """
    if not isinstance(data, dict):
        return False
    if not set(data) <= _ENVELOPE_KEYS or "model" not in data:
        return False
    model_json = data["model"]
    if not isinstance(model_json, str):
        return False
    raw_json = data.get("raw")
    if raw_json is not None and not isinstance(raw_json, str):
        return False
    try:
        json.loads(model_json)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def load_cached_response(
    cache: BaseCache,
    key: str,
    response_model: type[BaseModel],
    *,
    context: dict[str, Any] | None = None,
    strict: bool | None = None,
):
    """Return parsed model if *key* exists in *cache* else None."""
    reject_async_validators(response_model)
    cached = cache.get(key)
    if cached is None:
        return None

    model_json = cached
    raw_json = None
    try:
        data = json.loads(cached)
    except Exception:
        # Not JSON at all: the whole cached value is the model's JSON.
        pass
    else:
        if _is_response_envelope(data):
            model_json = data["model"]
            raw_json = data.get("raw")
        # Otherwise the cached document is the model's own JSON (legacy).

    obj = response_model.model_validate_json(model_json, context=context, strict=strict)
    if raw_json is not None:
        # `_raw_response` is an internal attribute used by Instructor; it may not
        # be declared on the Pydantic model type.
        try:
            raw_data = json.loads(raw_json)

            if isinstance(raw_data, dict) and any(
                key in raw_data for key in ["id", "object", "model", "choices"]
            ):
                # Completion-like mappings regain attribute access; other JSON
                # shapes remain plain data structures.
                object.__setattr__(
                    obj,
                    "_raw_response",
                    json.loads(raw_json, object_hook=lambda d: SimpleNamespace(**d)),
                )
                logger.debug("Restored raw response as SimpleNamespace object")
            else:
                object.__setattr__(obj, "_raw_response", raw_data)
                logger.debug("Restored raw response as plain data structure")
        except (json.JSONDecodeError, TypeError):
            # Preserve the historical string fallback.
            object.__setattr__(obj, "_raw_response", raw_json)
            logger.debug(
                "Restored raw response as string (original could not be fully serialized)"
            )
    logger.debug("cache hit: %s", key)
    return obj


def store_cached_response(
    cache: BaseCache, key: str, model: BaseModel, ttl: int | None = None
) -> None:
    """Serialize *model* and optional raw response to JSON and cache it."""
    raw_resp = getattr(model, "_raw_response", None)
    if raw_resp is not None:
        try:
            # Try Pydantic model serialization first (OpenAI, Anthropic, etc.)
            raw_resp_dump = getattr(raw_resp, "model_dump_json", None)
            if callable(raw_resp_dump):
                raw_json = raw_resp_dump()
            else:
                raise AttributeError("raw_resp has no model_dump_json")
            logger.debug("Cached raw response as Pydantic JSON")
        except (AttributeError, TypeError):
            # Fallback for non-Pydantic responses (custom providers, plain dicts, etc.)
            try:
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
        "model": model.model_dump_json(round_trip=True),
        "raw": raw_json,
    }
    cache.set(key, json.dumps(payload), ttl=ttl)
    logger.debug("cache store: %s", key)
