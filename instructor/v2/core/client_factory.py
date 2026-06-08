"""Shared native-client validation and Instructor construction."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from functools import cache
from typing import Any

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import ClientError, ModeError
from instructor.v2.core.mode import Mode
from instructor.v2.core.patch import patch_v2
from instructor.v2.core.provider_specs import PROVIDER_SPECS, ClientSpec
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry, normalize_mode


@cache
def _resolve_type(path: str) -> type[Any]:
    module_name, name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    if not isinstance(value, type):
        raise TypeError(f"{path} does not resolve to a type")
    return value


def _resolve_types(paths: tuple[str, ...], message: str) -> tuple[type[Any], ...]:
    try:
        return tuple(_resolve_type(path) for path in paths)
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise ClientError(message) from exc


def _resolve_method(client: Any, path: str) -> Callable[..., Any]:
    value = client
    try:
        for part in path.split("."):
            value = getattr(value, part)
    except AttributeError as exc:
        raise ClientError(
            f"Client {type(client).__name__} does not provide method {path!r}"
        ) from exc
    if not callable(value):
        raise ClientError(f"Client method {path!r} is not callable")
    return value


def _validate_mode(provider: Provider, mode: Mode) -> Mode:
    normalized = normalize_mode(provider, mode)
    if mode_registry.is_registered(provider, normalized):
        return normalized
    raise ModeError(
        mode=str(mode.value),
        provider=str(provider.value),
        valid_modes=[
            str(item.value) for item in mode_registry.get_modes_for_provider(provider)
        ],
    )


def _client_error(types: tuple[type[Any], ...], client: Any) -> ClientError:
    names = ", ".join(item.__name__ for item in types)
    return ClientError(
        f"Client must be an instance of one of: {names}. Got: {type(client).__name__}"
    )


def _stream_switch(
    create: Callable[..., Any],
    stream: Callable[..., Any] | None,
    *,
    is_async: bool,
    default_model: str | None = None,
    falsey_model_fallback: bool = False,
) -> Callable[..., Any]:
    if stream is None and not is_async and not falsey_model_fallback:
        return create

    def prepare(kwargs: dict[str, Any]) -> None:
        if falsey_model_fallback and not kwargs.get("model"):
            kwargs["model"] = default_model or ""

    if is_async:

        async def async_create(*args: Any, **kwargs: Any) -> Any:
            prepare(kwargs)
            wants_stream = kwargs.pop("stream", False) if stream is not None else False
            selected = stream if wants_stream and stream is not None else create
            result = selected(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        return async_create

    def sync_create(*args: Any, **kwargs: Any) -> Any:
        prepare(kwargs)
        if kwargs.pop("stream", False):
            assert stream is not None
            return stream(*args, **kwargs)
        return create(*args, **kwargs)

    return sync_create


def create_instructor(
    client: Any,
    *,
    provider: Provider,
    mode: Mode,
    model: str | None = None,
    use_async: bool | None = None,
    create_path: str | None = None,
    async_create_path: str | None = None,
    stream_path: str | None = None,
    async_stream_path: str | None = None,
    sync_types: tuple[type[Any], ...] | None = None,
    async_types: tuple[type[Any], ...] | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Build a sync or async wrapper from a provider's declarative client spec."""
    provider_spec = PROVIDER_SPECS[provider]
    contract = provider_spec.client
    if contract is None:
        raise ClientError(f"Provider {provider.value} has no native client contract")

    message = provider_spec.missing_sdk_message or (
        f"{provider_spec.sdk_module or provider.value} is not installed"
    )
    if contract.validation_order == "mode-first":
        normalized_mode = _validate_mode(provider, mode)
    resolved_sync_types = sync_types or _resolve_types(contract.sync_types, message)
    resolved_async_types = async_types or (
        _resolve_types(contract.async_types, message) if contract.async_types else ()
    )
    if contract.validation_order == "dependency-first":
        normalized_mode = _validate_mode(provider, mode)
    valid_types = (*resolved_sync_types, *resolved_async_types)
    if not isinstance(client, valid_types):
        raise _client_error(valid_types, client)
    if contract.validation_order == "client-first":
        normalized_mode = _validate_mode(provider, mode)

    is_async = (
        use_async
        if use_async is not None
        else bool(resolved_async_types and isinstance(client, resolved_async_types))
    )

    if is_async:
        selected_create_path = (
            async_create_path or create_path or contract.async_create or contract.create
        )
        selected_stream_path = async_stream_path or stream_path or contract.async_stream
    else:
        selected_create_path = create_path or contract.create
        selected_stream_path = stream_path or contract.stream
    create = _resolve_method(client, selected_create_path)
    stream = (
        _resolve_method(client, selected_stream_path)
        if selected_stream_path is not None
        else None
    )
    patched = patch_v2(
        func=_stream_switch(
            create,
            stream,
            is_async=is_async,
            default_model=model,
            falsey_model_fallback=contract.falsey_model_fallback,
        ),
        provider=provider,
        mode=normalized_mode,
        default_model=model,
    )
    wrapper = AsyncInstructor if is_async else Instructor
    return wrapper(
        client=client,
        create=patched,
        provider=provider,
        mode=normalized_mode,
        **kwargs,
    )


__all__ = ["ClientSpec", "create_instructor"]
