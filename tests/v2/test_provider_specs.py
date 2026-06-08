"""Provider manifest invariants."""

from __future__ import annotations

import warnings

import pytest

from instructor import Mode
from instructor import Provider
from instructor.v2.auto_client import supported_providers
from instructor.v2.core.provider_specs import ALIAS_TO_PROVIDER, PROVIDER_SPECS
from instructor.v2.core.providers import (
    normalize_mode_for_provider,
    provider_from_mode,
)
from tests.v2.provider_matrix import (
    EXPLICIT_PARALLEL_PROVIDERS,
    TYPED_MULTIMODAL_PROVIDERS,
)


def test_supported_provider_aliases_come_from_manifest() -> None:
    assert set(supported_providers) == set(ALIAS_TO_PROVIDER)


def test_supported_provider_aliases_have_auto_client_builders() -> None:
    routed_aliases = {
        alias
        for alias, provider in ALIAS_TO_PROVIDER.items()
        if PROVIDER_SPECS[provider].model_builder_module is not None
    }
    assert set(supported_providers) <= routed_aliases


def test_compatibility_aliases_point_to_canonical_providers() -> None:
    assert PROVIDER_SPECS[Provider.GENERATIVE_AI].canonical_provider is Provider.GENAI
    assert PROVIDER_SPECS[Provider.AZURE_OPENAI].canonical_provider is Provider.OPENAI
    assert PROVIDER_SPECS[Provider.OLLAMA].canonical_provider is Provider.OPENAI


def test_first_class_specs_are_self_canonical() -> None:
    for spec in PROVIDER_SPECS.values():
        if spec.provider in {
            Provider.GENERATIVE_AI,
            Provider.AZURE_OPENAI,
            Provider.OLLAMA,
        }:
            continue
        assert spec.canonical_provider is spec.provider


@pytest.mark.parametrize(
    "spec",
    [spec for spec in PROVIDER_SPECS.values() if spec.handler_module is not None],
    ids=lambda spec: spec.provider.value,
)
def test_advertised_streaming_modes_are_supported_modes(spec) -> None:
    advertised_modes = {
        *spec.capabilities.partial_stream_modes,
        *spec.capabilities.iterable_stream_modes,
    }
    assert advertised_modes <= set(spec.supported_modes)


@pytest.mark.parametrize(
    "spec",
    [spec for spec in PROVIDER_SPECS.values() if spec.handler_module is not None],
    ids=lambda spec: spec.provider.value,
)
def test_explicit_parallel_contract_matches_supported_mode(spec) -> None:
    assert spec.capabilities.explicit_parallel_tools is (
        Mode.PARALLEL_TOOLS in spec.supported_modes
    )


def test_known_public_streaming_gaps_are_not_advertised() -> None:
    assert (
        Mode.MD_JSON
        not in PROVIDER_SPECS[Provider.XAI].capabilities.partial_stream_modes
    )
    assert not PROVIDER_SPECS[Provider.GEMINI].capabilities.iterable_stream_modes
    assert not PROVIDER_SPECS[Provider.VERTEXAI].capabilities.iterable_stream_modes


def test_multimodal_contract_is_explicitly_typed_media_only() -> None:
    assert Provider.GENAI in TYPED_MULTIMODAL_PROVIDERS
    assert Provider.BEDROCK not in TYPED_MULTIMODAL_PROVIDERS
    assert PROVIDER_SPECS[Provider.ANTHROPIC].capabilities.multimodal_inputs == (
        "image",
        "pdf",
    )


def test_explicit_parallel_contract_is_driven_by_manifest() -> None:
    assert set(EXPLICIT_PARALLEL_PROVIDERS) == {
        provider
        for provider, spec in PROVIDER_SPECS.items()
        if spec.handler_module is not None
        and Mode.PARALLEL_TOOLS in spec.supported_modes
    }


def test_legacy_mode_ownership_and_normalization_come_from_manifest() -> None:
    owners: dict[Mode, list[Provider]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for provider, spec in PROVIDER_SPECS.items():
            for legacy_mode, canonical_mode in spec.legacy_modes.items():
                owners.setdefault(legacy_mode, []).append(provider)
                assert (
                    normalize_mode_for_provider(legacy_mode, provider) is canonical_mode
                )

    for legacy_mode, providers in owners.items():
        canonical_owners = {
            PROVIDER_SPECS[provider].canonical_provider for provider in providers
        }
        if len(canonical_owners) == 1:
            assert provider_from_mode(legacy_mode) is canonical_owners.pop()
        else:
            assert provider_from_mode(legacy_mode, Provider.COHERE) is Provider.COHERE
