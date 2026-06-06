from __future__ import annotations

from types import SimpleNamespace

import pytest

from instructor.mode import Mode
from instructor.utils.providers import Provider

try:
    from instructor.v2 import from_genai
    from instructor.v2.core import mode_registry
    from google.genai import types
except ModuleNotFoundError:
    # fmt: off
    pytest.skip("google-genai package is not installed", allow_module_level=True)  # ty: ignore[too-many-positional-arguments]
    # fmt: on


class DummyModels:
    def __init__(self):
        self.called = False
        self.stream_called = False

    def generate_content(self, *_args, **_kwargs):
        self.called = True
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[]))]
        )

    def generate_content_stream(self, *_args, **_kwargs):
        self.stream_called = True
        yield types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[]))]
        )


class DummyAsyncModels:
    def __init__(self):
        self.called = False

    async def generate_content(self, *_args, **_kwargs):
        self.called = True
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[]))]
        )

    async def generate_content_stream(self, *_args, **_kwargs):
        self.called = True

        async def _gen():
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(content=types.Content(role="model", parts=[]))
                ]
            )

        return _gen()


class DummyClient:
    def __init__(self):
        self.models = DummyModels()
        self.aio = SimpleNamespace(models=DummyAsyncModels())


def test_mode_registry_has_genai_handlers():
    # Test generic modes
    assert mode_registry.is_registered(Provider.GENAI, Mode.TOOLS)
    assert mode_registry.is_registered(Provider.GENAI, Mode.JSON)
    # Legacy modes remain accepted through the v2 normalization shim.
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_TOOLS)
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_JSON)
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS)


def test_from_genai_sync_generic_mode(monkeypatch):
    """Test using generic Mode.TOOLS."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )

    client = DummyClient()
    instructor = from_genai(client, mode=Mode.TOOLS, use_async=False)
    instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )

    assert client.models.called


def test_from_genai_sync_legacy_mode_normalized(monkeypatch):
    """Legacy Mode.GENAI_TOOLS remains accepted in v2."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )

    client = DummyClient()
    instructor = from_genai(client, mode=Mode.GENAI_TOOLS, use_async=False)
    instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )
    assert client.models.called


@pytest.mark.asyncio
async def test_from_genai_async_generic_mode(monkeypatch):
    """Test using generic Mode.TOOLS with async."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )
    client = DummyClient()
    instructor = from_genai(client, mode=Mode.TOOLS, use_async=True)
    await instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )
    assert client.aio.models.called


@pytest.mark.asyncio
async def test_from_genai_async_legacy_mode_normalized(monkeypatch):
    """Legacy Mode.GENAI_TOOLS remains accepted in async v2."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )
    client = DummyClient()
    instructor = from_genai(client, mode=Mode.GENAI_TOOLS, use_async=True)
    await instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )
    assert client.aio.models.called


def test_from_genai_json_mode(monkeypatch):
    """Test using generic Mode.JSON."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )

    client = DummyClient()
    instructor = from_genai(client, mode=Mode.JSON, use_async=False)
    instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )

    assert client.models.called


def test_from_genai_json_legacy_mode_normalized(monkeypatch):
    """Legacy structured outputs mode remains accepted in v2."""
    monkeypatch.setattr(
        "instructor.v2.providers.genai.client.Client",
        DummyClient,
    )

    client = DummyClient()
    instructor = from_genai(
        client,
        mode=Mode.GENAI_STRUCTURED_OUTPUTS,
        use_async=False,
    )
    instructor.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        response_model=None,
    )
    assert client.models.called
