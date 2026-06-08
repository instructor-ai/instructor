from __future__ import annotations

import pytest

from instructor.v2.core.multimodal import Image, PDF, PDFWithCacheControl
from instructor.v2.providers.anthropic import handlers as anthropic_handlers
from instructor.v2.providers.mistral import handlers as mistral_handlers
from instructor.v2.providers.openai import handlers as openai_handlers


def _image() -> Image:
    return Image(
        source="data:image/png;base64,AA==",
        media_type="image/png",
        data="AA==",
    )


def test_openai_handler_uses_openai_media_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        openai_handlers,
        "media_to_openai",
        lambda _media, mode: {"provider": "openai", "mode": mode.value},
    )

    converted = openai_handlers.OpenAIToolsHandler().convert_messages(
        [{"role": "user", "content": [_image()]}]
    )

    assert converted[0]["content"] == [{"provider": "openai", "mode": "tool_call"}]


def test_anthropic_handler_uses_anthropic_media_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        anthropic_handlers,
        "media_to_anthropic",
        lambda _media: {"provider": "anthropic"},
    )

    converted = anthropic_handlers.AnthropicToolsHandler().convert_messages(
        [{"role": "user", "content": [_image()]}]
    )

    assert converted[0]["content"] == [{"provider": "anthropic"}]


def test_mistral_handler_uses_mistral_media_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mistral_handlers,
        "media_to_mistral",
        lambda _media, mode: {"provider": "mistral", "mode": mode.value},
    )

    converted = mistral_handlers.MistralToolsHandler().convert_messages(
        [{"role": "user", "content": [_image()]}]
    )

    assert converted[0]["content"] == [{"provider": "mistral", "mode": "mistral_tools"}]


@pytest.mark.parametrize(
    ("handlers", "handler_cls", "factory_name"),
    [
        (openai_handlers, openai_handlers.OpenAIToolsHandler, "image_from_params"),
        (
            anthropic_handlers,
            anthropic_handlers.AnthropicToolsHandler,
            "image_from_params",
        ),
        (mistral_handlers, mistral_handlers.MistralToolsHandler, "image_from_params"),
    ],
)
def test_active_handlers_own_image_shorthand_conversion(
    monkeypatch: pytest.MonkeyPatch,
    handlers: object,
    handler_cls: type[object],
    factory_name: str,
) -> None:
    image = _image()
    monkeypatch.setattr(handlers, factory_name, lambda _params: image)

    converted = handler_cls().convert_messages(  # type: ignore[attr-defined]
        [{"role": "user", "content": [{"type": "image", "source": "ignored"}]}],
        autodetect_images=True,
    )

    assert converted[0]["content"]


def test_anthropic_cache_compatibility_is_data_driven() -> None:
    ordinary_pdf = PDF(
        source="data:application/pdf;base64,AA==",
        media_type="application/pdf",
        data="AA==",
    )
    cacheable_pdf = PDFWithCacheControl(
        source=ordinary_pdf.source,
        media_type=ordinary_pdf.media_type,
        data=ordinary_pdf.data,
    )

    assert "cache_control" not in anthropic_handlers.media_to_anthropic(ordinary_pdf)
    assert anthropic_handlers.media_to_anthropic(cacheable_pdf)["cache_control"] == {
        "type": "ephemeral"
    }
