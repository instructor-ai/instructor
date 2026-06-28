"""Wiring tests for the OrcaRouter provider.

These tests do not hit the live API. They verify that the provider is
registered correctly across the enum, mode, handler-registry and
``from_provider`` routing layers. Live integration tests can be added in
``test_live.py`` once ``ORCAROUTER_API_KEY`` is available in CI.
"""

from __future__ import annotations

import os

import pytest

import instructor
from instructor import Mode, Provider
from instructor.v2.core.providers import get_provider, provider_from_mode


def test_provider_enum_has_orcarouter() -> None:
    assert Provider.ORCAROUTER.value == "orcarouter"


def test_mode_has_orcarouter_structured_outputs() -> None:
    assert Mode.ORCAROUTER_STRUCTURED_OUTPUTS.value == "orcarouter_structured_outputs"
    assert Mode.ORCAROUTER_STRUCTURED_OUTPUTS in Mode.tool_modes()
    assert Mode.ORCAROUTER_STRUCTURED_OUTPUTS in Mode.json_modes()


def test_provider_from_mode_maps_to_orcarouter() -> None:
    assert (
        provider_from_mode(Mode.ORCAROUTER_STRUCTURED_OUTPUTS) is Provider.ORCAROUTER
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://api.orcarouter.ai/v1",
        "https://www.orcarouter.ai",
        "https://orcarouter.ai/v1",
    ],
)
def test_get_provider_detects_orcarouter(url: str) -> None:
    assert get_provider(url) is Provider.ORCAROUTER


def test_from_orcarouter_is_exported() -> None:
    assert hasattr(instructor, "from_orcarouter")
    assert callable(instructor.from_orcarouter)


def test_from_provider_routes_to_orcarouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    client = instructor.from_provider(
        "orcarouter/openai/gpt-4o-mini",
        async_client=False,
    )
    assert client.provider is Provider.ORCAROUTER
    assert client.mode is Mode.TOOLS
    assert str(client.client.base_url).startswith("https://api.orcarouter.ai")


def test_from_provider_async_routes_to_orcarouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    client = instructor.from_provider(
        "orcarouter/openai/gpt-4o-mini",
        async_client=True,
    )
    assert client.provider is Provider.ORCAROUTER


def test_from_provider_missing_api_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    from instructor.v2.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        instructor.from_provider("orcarouter/openai/gpt-4o-mini")


def test_from_provider_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    client = instructor.from_provider(
        "orcarouter/openai/gpt-4o-mini",
        base_url="https://custom.orcarouter.example/v1",
    )
    assert "custom.orcarouter.example" in str(client.client.base_url)


def test_orcarouter_supports_json_schema_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    client = instructor.from_provider(
        "orcarouter/openai/gpt-4o-mini",
        mode=Mode.JSON_SCHEMA,
    )
    assert client.mode is Mode.JSON_SCHEMA


def test_orcarouter_legacy_mode_normalises_to_json_schema() -> None:
    """ORCAROUTER_STRUCTURED_OUTPUTS is a legacy alias for JSON_SCHEMA."""
    from instructor.v2.core.mode import DEPRECATED_TO_CORE

    assert DEPRECATED_TO_CORE[Mode.ORCAROUTER_STRUCTURED_OUTPUTS] is Mode.JSON_SCHEMA


@pytest.mark.skipif(
    not os.getenv("ORCAROUTER_API_KEY"),
    reason="ORCAROUTER_API_KEY not set; skipping live integration test",
)
def test_live_orcarouter_simple_extraction() -> None:
    """Smoke test: end-to-end extraction against the OrcaRouter API."""
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_provider(
        "orcarouter/openai/gpt-4o-mini",
        mode=Mode.TOOLS,
    )
    resp = client.create(
        messages=[
            {
                "role": "user",
                "content": "Extract the user: Ivan is 27 and lives in Singapore.",
            }
        ],
        response_model=User,
    )
    assert resp.name.lower() == "ivan"
    assert resp.age == 27
