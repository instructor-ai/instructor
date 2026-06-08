from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from instructor import Mode, Provider
from instructor.v2.core.multimodal import Audio, Image, PDF
from instructor.v2.core.provider_specs import PROVIDER_SPECS
from instructor.v2.core.registry import mode_registry
from tests.v2.provider_matrix import (
    ITERABLE_STREAM_CASES,
    PARTIAL_STREAM_CASES,
    TYPED_MULTIMODAL_CASES,
    ensure_handlers_loaded,
)

_PAYLOAD = '{"answer": 4}'
_STREAM_CASES = tuple(dict.fromkeys((*PARTIAL_STREAM_CASES, *ITERABLE_STREAM_CASES)))


class _GeminiFunctionCall:
    @staticmethod
    def to_dict(_value: Any) -> dict[str, dict[str, int]]:
        return {"args": {"answer": 4}}


def _stream_chunk(provider: Provider, mode: Mode) -> Any:
    module = PROVIDER_SPECS[provider].handler_module
    if module == "instructor.v2.providers.anthropic.handlers":
        return SimpleNamespace(
            delta=SimpleNamespace(partial_json=_PAYLOAD, text=_PAYLOAD)
        )
    if module == "instructor.v2.providers.genai.handlers":
        part = SimpleNamespace(function_call=SimpleNamespace(args={"answer": 4}))
        return SimpleNamespace(
            text=_PAYLOAD,
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
        )
    if module == "instructor.v2.providers.gemini.handlers":
        part = SimpleNamespace(function_call=_GeminiFunctionCall())
        return SimpleNamespace(
            text=_PAYLOAD,
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
        )
    if module == "instructor.v2.providers.vertexai.handlers":
        part = SimpleNamespace(
            text=_PAYLOAD,
            function_call=SimpleNamespace(args={"answer": 4}),
        )
        return SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]
        )
    if module == "instructor.v2.providers.cohere.handlers":
        return SimpleNamespace(text=_PAYLOAD)
    if module == "instructor.v2.providers.mistral.handlers":
        delta = SimpleNamespace(
            content=_PAYLOAD,
            tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments=_PAYLOAD))],
        )
        return SimpleNamespace(
            data=SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
        )
    if mode is Mode.RESPONSES_TOOLS:
        from openai.types.responses import ResponseFunctionCallArgumentsDeltaEvent

        return ResponseFunctionCallArgumentsDeltaEvent.model_validate(
            {
                "delta": _PAYLOAD,
                "item_id": "call_1",
                "output_index": 0,
                "sequence_number": 0,
                "type": "response.function_call_arguments.delta",
            }
        )
    delta = SimpleNamespace(
        content=_PAYLOAD,
        tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments=_PAYLOAD))],
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


@pytest.mark.parametrize(("provider", "mode"), _STREAM_CASES)
def test_advertised_stream_contract_extracts_payload(
    provider: Provider, mode: Mode
) -> None:
    ensure_handlers_loaded(provider, skip_missing_dependency=True)
    extractor = mode_registry.get_handlers(provider, mode).stream_extractor

    assert extractor is not None
    assert "answer" in "".join(extractor([_stream_chunk(provider, mode)]))


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "mode"), _STREAM_CASES)
async def test_advertised_async_stream_contract_extracts_payload(
    provider: Provider, mode: Mode
) -> None:
    ensure_handlers_loaded(provider, skip_missing_dependency=True)
    extractor = mode_registry.get_handlers(provider, mode).stream_extractor_async

    async def chunks():
        yield _stream_chunk(provider, mode)

    assert extractor is not None
    extracted: Any = extractor(chunks())
    if inspect.isawaitable(extracted):
        extracted = await extracted
    assert "answer" in "".join([part async for part in extracted])


def _media(provider: Provider, media_type: str) -> Image | Audio | PDF:
    if media_type == "image":
        return Image(
            source="data:image/png;base64,AA==", media_type="image/png", data="AA=="
        )
    if media_type == "audio":
        return Audio(
            source="data:audio/wav;base64,AA==", media_type="audio/wav", data="AA=="
        )
    if provider is Provider.MISTRAL:
        return PDF(source="https://example.com/document.pdf", data=None)
    return PDF(source="data:application/pdf;base64,AA==", data="AA==")


@pytest.mark.parametrize(("provider", "media_type"), TYPED_MULTIMODAL_CASES)
def test_advertised_multimodal_contract_dispatches_typed_media(
    provider: Provider, media_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = _media(provider, media_type)
    if provider is Provider.GENAI:
        from instructor.v2.providers.genai import multimodal

        part = SimpleNamespace(
            from_bytes=lambda **kwargs: kwargs,
            from_uri=lambda **kwargs: kwargs,
        )
        monkeypatch.setattr(multimodal, "_types", lambda: SimpleNamespace(Part=part))
        assert multimodal.media_to_genai(media)
        return

    ensure_handlers_loaded(provider, skip_missing_dependency=True)
    modes = PROVIDER_SPECS[provider].supported_modes
    mode = Mode.TOOLS if Mode.TOOLS in modes else modes[0]
    converter = mode_registry.get_handlers(provider, mode).message_converter

    assert converter is not None
    converted = converter([{"role": "user", "content": [media]}])
    assert converted[0]["content"]
