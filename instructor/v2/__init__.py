"""Instructor v2 public exports with lazy loading."""

from __future__ import annotations

from importlib import import_module
from typing import Any
from instructor.v2.core.provider_specs import PUBLIC_FACTORY_ATTRS

__all__ = [
    "Mode",
    "Provider",
    "ModeHandler",
    "ModeHandlers",
    "ModeRegistry",
    "mode_registry",
    "normalize_mode",
    "patch_v2",
    "register_mode_handler",
    "ReaskHandler",
    "RequestHandler",
    "ResponseParser",
    "providers",
    *PUBLIC_FACTORY_ATTRS.keys(),
]

_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    "Mode": ("instructor.v2.core.mode", "Mode"),
    "Provider": ("instructor.v2.core.providers", "Provider"),
    "ModeHandler": ("instructor.v2.core.handler", "ModeHandler"),
    "ModeHandlers": ("instructor.v2.core.registry", "ModeHandlers"),
    "ModeRegistry": ("instructor.v2.core.registry", "ModeRegistry"),
    "mode_registry": ("instructor.v2.core.registry", "mode_registry"),
    "normalize_mode": ("instructor.v2.core.registry", "normalize_mode"),
    "patch_v2": ("instructor.v2.core.patch", "patch_v2"),
    "register_mode_handler": (
        "instructor.v2.core.decorators",
        "register_mode_handler",
    ),
    "ReaskHandler": ("instructor.v2.core.protocols", "ReaskHandler"),
    "RequestHandler": ("instructor.v2.core.protocols", "RequestHandler"),
    "ResponseParser": ("instructor.v2.core.protocols", "ResponseParser"),
    "providers": ("instructor.v2.providers", None),
    **PUBLIC_FACTORY_ATTRS,
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_path)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
