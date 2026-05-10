"""Compatibility tests for deprecated ResponseSchema provider helpers."""

from __future__ import annotations

from typing import Any

import pytest

from instructor import Mode, Provider
from instructor.v2.core.function_calls import ResponseSchema


class Answer(ResponseSchema):
    answer: float


@pytest.mark.parametrize(
    ("method_name", "mode", "provider"),
    [
        ("parse_genai_structured_outputs", Mode.JSON, Provider.GENAI),
        ("parse_genai_tools", Mode.TOOLS, Provider.GENAI),
        ("parse_cohere_json_schema", Mode.JSON_SCHEMA, Provider.COHERE),
        ("parse_bedrock_json", Mode.MD_JSON, Provider.BEDROCK),
        ("parse_bedrock_tools", Mode.TOOLS, Provider.BEDROCK),
        ("parse_gemini_json", Mode.MD_JSON, Provider.GEMINI),
        ("parse_gemini_tools", Mode.TOOLS, Provider.GEMINI),
        ("parse_vertexai_tools", Mode.TOOLS, Provider.VERTEXAI),
        ("parse_vertexai_json", Mode.MD_JSON, Provider.VERTEXAI),
        ("parse_cohere_tools", Mode.TOOLS, Provider.COHERE),
        ("parse_writer_tools", Mode.TOOLS, Provider.WRITER),
        ("parse_writer_json", Mode.MD_JSON, Provider.WRITER),
        ("parse_mistral_structured_outputs", Mode.JSON_SCHEMA, Provider.MISTRAL),
    ],
)
def test_provider_parse_helpers_delegate_to_registry(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    mode: Mode,
    provider: Provider,
) -> None:
    calls: list[dict[str, Any]] = []
    sentinel = object()

    def fake_parse_with_registry(
        cls: type[ResponseSchema],
        completion: Any,
        *,
        mode: Mode,
        provider: Provider,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        warning: str | None = None,
    ) -> object:
        calls.append(
            {
                "cls": cls,
                "completion": completion,
                "mode": mode,
                "provider": provider,
                "validation_context": validation_context,
                "strict": strict,
                "warning": warning,
            }
        )
        return sentinel

    monkeypatch.setattr(
        Answer,
        "_parse_with_registry",
        classmethod(fake_parse_with_registry),
    )

    completion = object()
    kwargs: dict[str, Any] = {"validation_context": {"source": "compat"}}
    if method_name != "parse_vertexai_tools":
        kwargs["strict"] = True
    result = getattr(Answer, method_name)(completion, **kwargs)

    assert result is sentinel
    assert calls == [
        {
            "cls": Answer,
            "completion": completion,
            "mode": mode,
            "provider": provider,
            "validation_context": {"source": "compat"},
            "strict": False if method_name == "parse_vertexai_tools" else True,
            "warning": calls[0]["warning"],
        }
    ]
    assert calls[0]["warning"] is not None
