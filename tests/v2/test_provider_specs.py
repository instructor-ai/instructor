"""Provider manifest invariants."""

from __future__ import annotations

from instructor import Provider
from instructor.v2.auto_client import supported_providers
from instructor.v2.core.provider_specs import ALIAS_TO_PROVIDER, PROVIDER_SPECS


def test_supported_provider_aliases_come_from_manifest() -> None:
    assert set(supported_providers) == set(ALIAS_TO_PROVIDER)


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
