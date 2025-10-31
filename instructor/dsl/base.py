"""Shared interfaces for Instructor DSL response models."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Callable, ClassVar

from ..mode import Mode

ProviderHook = Callable[[Any, dict[str, Any]], tuple[Any, dict[str, Any]]]


class DSLProviderHooksMixin:
    """Mixin that standardizes provider hook behaviour for DSL models."""

    provider_mode_hooks: ClassVar[dict[Mode, ProviderHook]] = {}
    stream_consumer: ClassVar[bool] = False

    @classmethod
    def apply_provider_hook(
        cls,
        mode: Mode,
        kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Apply a provider-specific hook if one is registered."""

        hook = cls.provider_mode_hooks.get(mode)
        if hook is None:
            return cls, kwargs
        return hook(cls, kwargs)

    @classmethod
    def consume_stream_sync(
        cls,
        completion: Any,
        mode: Mode,
    ) -> list[Any]:
        """Collect streaming results for synchronous clients."""

        raise NotImplementedError(f"{cls.__name__} does not support streaming consumption")

    @classmethod
    async def consume_stream_async(
        cls,
        completion: AsyncGenerator[Any, None],
        mode: Mode,
    ) -> list[Any]:
        """Collect streaming results for async clients."""

        raise NotImplementedError(f"{cls.__name__} does not support streaming consumption")

    @classmethod
    def finalize_response(cls, model: Any, raw_response: Any) -> Any:
        """Finalize a parsed response before returning it to the caller."""

        model._raw_response = raw_response  # type: ignore[attr-defined]
        return model


def get_dsl_provider_hooks(
    response_model: Any,
) -> type[DSLProviderHooksMixin] | None:
    """Return the hook mixin for a response model when available."""

    if isinstance(response_model, type) and issubclass(
        response_model, DSLProviderHooksMixin
    ):
        return response_model

    if isinstance(response_model, DSLProviderHooksMixin):
        return response_model.__class__

    return None


__all__ = ["DSLProviderHooksMixin", "ProviderHook", "get_dsl_provider_hooks"]
