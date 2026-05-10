"""Tests for shared-core dispatch into provider-owned helpers."""

from __future__ import annotations

from typing import Any

import pytest

from instructor import Mode, Provider
from instructor.v2.core import retry, templating


@pytest.mark.parametrize(
    ("provider", "message", "target"),
    [
        (
            Provider.OPENAI,
            {"content": "Hello {{ name }}"},
            "instructor.v2.providers.openai.templating.process_message",
        ),
        (
            Provider.ANTHROPIC,
            {"content": [{"type": "text", "text": "Hello {{ name }}"}]},
            "instructor.v2.providers.anthropic.templating.process_message",
        ),
        (
            Provider.GEMINI,
            {"parts": ["Hello {{ name }}"]},
            "instructor.v2.providers.gemini.templating.process_message",
        ),
        (
            Provider.COHERE,
            {"message": "Hello {{ name }}"},
            "instructor.v2.providers.cohere.templating.process_message",
        ),
    ],
)
def test_process_message_dispatches_to_provider_modules(
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    message: dict[str, Any],
    target: str,
) -> None:
    calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def fake_process_message(
        value: dict[str, Any],
        context: dict[str, Any],
        _apply_template: Any,
    ) -> dict[str, str]:
        calls.append((value, context))
        return {"provider": provider.value}

    monkeypatch.setattr(target, fake_process_message)

    assert templating.process_message(message, {"name": "Ada"}, provider) == {
        "provider": provider.value
    }
    assert calls == [(message, {"name": "Ada"})]


def test_initialize_usage_dispatches_anthropic_to_provider_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[str] = []

    def fake_initialize_usage() -> object:
        calls.append("anthropic")
        return sentinel

    monkeypatch.setattr(
        "instructor.v2.providers.anthropic.usage.initialize_usage",
        fake_initialize_usage,
    )

    assert retry._initialize_usage(Mode.ANTHROPIC_TOOLS) is sentinel
    assert calls == ["anthropic"]
