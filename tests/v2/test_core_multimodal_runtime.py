from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from instructor import Mode
from instructor.v2.core.multimodal import (
    Audio,
    Image,
    ImageWithCacheControl,
    PDF,
    autodetect_media,
    convert_contents,
    convert_messages,
)


def test_convert_contents_uses_responses_text_shape() -> None:
    assert convert_contents("hello", Mode.RESPONSES_TOOLS) == "hello"
    assert convert_contents(["hello"], Mode.RESPONSES_TOOLS) == [
        {"type": "input_text", "text": "hello"}
    ]


def test_convert_contents_routes_pdf_to_mistral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = PDF(
        source="https://example.com/doc.pdf",
        media_type="application/pdf",
        data=None,
    )
    monkeypatch.setattr(PDF, "to_mistral", lambda _self: {"target": "mistral"})
    monkeypatch.setattr(PDF, "to_openai", lambda _self, _mode: {"target": "openai"})

    assert convert_contents(pdf, Mode.MISTRAL_TOOLS) == [{"target": "mistral"}]


def test_convert_contents_rejects_unknown_object() -> None:
    with pytest.raises(ValueError, match="Unsupported content type"):
        convert_contents(
            [object()],  # ty: ignore[invalid-argument-type]
            Mode.TOOLS,
        )


@pytest.mark.parametrize(
    ("media_type", "autodetect"),
    [
        ("image", Image.autodetect),
        ("audio", Audio.autodetect),
        ("PDF", PDF.autodetect),
    ],
)
def test_autodetect_rejects_unsupported_source_types(
    media_type: str, autodetect: Any
) -> None:
    with pytest.raises(
        ValueError, match=rf"Unsupported {media_type} source type: bytes"
    ):
        autodetect(b"not a string or path")


def test_webp_mime_type_is_registered() -> None:
    import mimetypes

    assert mimetypes.guess_type("image.webp")[0] == "image/webp"


def test_autodetect_media_uses_mime_type_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image(source="file.png", media_type="image/png", data="AA==")
    monkeypatch.setattr("mimetypes.guess_type", lambda _: ("image/png", None))
    monkeypatch.setattr(Image, "autodetect_safely", lambda _source: image)

    def fail_audio_autodetect(_source: str) -> None:
        raise AssertionError("audio autodetect should not be called")

    monkeypatch.setattr(Audio, "autodetect_safely", fail_audio_autodetect)

    assert autodetect_media("picture.png") is image


def test_convert_messages_autodetects_image_params_and_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = ImageWithCacheControl(
        source="data:image/png;base64,AA==",
        media_type="image/png",
        data="AA==",
        cache_control={"type": "ephemeral"},
    )
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        ImageWithCacheControl,
        "from_image_params",
        classmethod(lambda _cls, _params: image),
    )
    monkeypatch.setattr(
        "instructor.v2.core.multimodal.autodetect_media",
        lambda value: f"detected:{value}",
    )

    def fake_convert_contents(contents: Any, mode: Mode) -> list[dict[str, str]]:
        seen["contents"] = contents
        seen["mode"] = mode
        return [{"type": "text", "text": "converted"}]

    monkeypatch.setattr(
        "instructor.v2.core.multimodal.convert_contents", fake_convert_contents
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                "photo.png",
                {"type": "image", "source": "data:image/png;base64,AA=="},
            ],
            "name": "tester",
        }
    ]

    converted = convert_messages(messages, Mode.TOOLS, autodetect_images=True)

    assert seen["contents"] == ["detected:photo.png", image]
    assert seen["mode"] == Mode.TOOLS
    assert converted == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "converted"}],
            "name": "tester",
        }
    ]


def test_convert_messages_rejects_unknown_typed_message() -> None:
    with pytest.raises(ValueError, match="Unsupported message type"):
        convert_messages(
            [{"type": "video", "role": "user", "content": "x"}], Mode.TOOLS
        )


def test_pdf_to_bedrock_uses_s3_source_and_sanitizes_name() -> None:
    pdf = PDF(source="s3://bucket/folder/My Report (Final)!.pdf", data=None)

    result = pdf.to_bedrock(name="My Report (Final)!.pdf")

    assert result["document"]["name"] == "My Report (Final)pdf"
    assert result["document"]["source"]["s3Location"]["uri"] == pdf.source


def test_pdf_to_bedrock_decodes_base64_bytes() -> None:
    encoded = base64.b64encode(b"%PDF-1.7 demo").decode("utf-8")
    pdf = PDF(source="data:application/pdf;base64," + encoded, data=encoded)

    result = pdf.to_bedrock(name="Doc")

    assert result["document"]["name"] == "Doc"
    assert result["document"]["source"]["bytes"] == b"%PDF-1.7 demo"


def test_pdf_to_bedrock_requires_data_for_non_s3_source() -> None:
    pdf = PDF(source="https://example.com/report.pdf", data=None)

    with pytest.raises(ValueError, match="PDF data is missing"):
        pdf.to_bedrock()


def test_audio_from_path_normalizes_windows_wav_and_aac_mime_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = tmp_path / "clip.wav"
    wav_path.write_bytes(b"RIFFdemo")
    aac_path = tmp_path / "clip.aac"
    aac_path.write_bytes(b"demo")

    mime_types = {
        str(wav_path): ("audio/x-wav", None),
        str(aac_path): ("audio/vnd.dlna.adts", None),
    }
    monkeypatch.setattr("mimetypes.guess_type", lambda path: mime_types[str(path)])

    wav_audio = Audio.from_path(wav_path)
    aac_audio = Audio.from_path(aac_path)

    assert wav_audio.media_type == "audio/wav"
    assert aac_audio.media_type == "audio/aac"
