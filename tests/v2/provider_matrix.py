"""Shared provider capability matrix for v2 tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from instructor import Provider
from instructor.v2.core.provider_specs import PROVIDER_SPECS
from instructor.v2.core.registry import mode_registry


TEST_PROVIDER_SPECS = {
    provider: spec
    for provider, spec in PROVIDER_SPECS.items()
    if spec.handler_module is not None and spec.from_function is not None
}


PROVIDER_HANDLER_MODES = {
    provider: spec.supported_modes for provider, spec in PROVIDER_SPECS.items()
}
PARTIAL_STREAM_CASES = tuple(
    (provider, mode)
    for provider, spec in TEST_PROVIDER_SPECS.items()
    for mode in spec.capabilities.partial_stream_modes
)
ITERABLE_STREAM_CASES = tuple(
    (provider, mode)
    for provider, spec in TEST_PROVIDER_SPECS.items()
    for mode in spec.capabilities.iterable_stream_modes
)
TYPED_MULTIMODAL_PROVIDERS = tuple(
    provider
    for provider, spec in TEST_PROVIDER_SPECS.items()
    if spec.capabilities.multimodal_inputs
)
TYPED_MULTIMODAL_CASES = tuple(
    (provider, media_type)
    for provider, spec in TEST_PROVIDER_SPECS.items()
    for media_type in spec.capabilities.multimodal_inputs
)
EXPLICIT_PARALLEL_PROVIDERS = tuple(
    provider
    for provider, spec in TEST_PROVIDER_SPECS.items()
    if spec.capabilities.explicit_parallel_tools
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HANDLERS_LOADED: set[Provider] = set()


def handler_module_path(provider: Provider) -> Path | None:
    """Return the registered handler implementation path for a provider."""
    module = PROVIDER_SPECS[provider].handler_module
    if module is None:
        return None
    return _PROJECT_ROOT / f"{module.replace('.', '/')}.py"


def _is_expected_missing_dependency(provider: Provider, exc: ImportError) -> bool:
    sdk_module = PROVIDER_SPECS[provider].sdk_module
    if sdk_module is None:
        return False
    expected_root = sdk_module.split(".")[0]
    missing_name = getattr(exc, "name", None)
    if missing_name:
        return missing_name.split(".")[0] == expected_root
    return f"No module named '{expected_root}'" in str(exc)


def ensure_handlers_loaded(
    provider: Provider, *, skip_missing_dependency: bool = False
) -> None:
    """Load handlers once from the manifest path so registration tests share setup."""
    if provider in _HANDLERS_LOADED:
        return
    provider_modes = PROVIDER_HANDLER_MODES.get(provider, ())
    if provider_modes and all(
        mode_registry.is_registered(provider, mode) for mode in provider_modes
    ):
        _HANDLERS_LOADED.add(provider)
        return
    path = handler_module_path(provider)
    if path is None or not path.exists():
        return
    spec = importlib.util.spec_from_file_location(
        f"tests.v2.handlers_{provider.value}",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load handler module for {provider}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError) as exc:
        if skip_missing_dependency and _is_expected_missing_dependency(provider, exc):
            pytest.skip(
                f"{provider.value} handlers require optional dependency "  # ty: ignore[too-many-positional-arguments]
                f"{PROVIDER_SPECS[provider].sdk_module}"
            )
        raise
    _HANDLERS_LOADED.add(provider)
