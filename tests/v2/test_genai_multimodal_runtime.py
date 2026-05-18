from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from instructor.v2.core.multimodal import Image, PDF, PDFWithGenaiFile


class FakePart:
    def __init__(
        self,
        *,
        data: bytes | None = None,
        mime_type: str | None = None,
        file_uri: str | None = None,
    ) -> None:
        self.data = data
        self.mime_type = mime_type
        self.file_uri = file_uri

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> FakePart:
        return cls(data=data, mime_type=mime_type)

    @classmethod
    def from_uri(cls, file_uri: str, mime_type: str) -> FakePart:
        return cls(file_uri=file_uri, mime_type=mime_type)


class FakeContent:
    def __init__(self, *, role: str, parts: list[Any]) -> None:
        self.role = role
        self.parts = parts


class FakeFile:
    pass


def _install_fake_genai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    client_factory: type[Any] | None = None,
) -> None:
    types_module = ModuleType("google.genai.types")
    types_module.Part = FakePart  # type: ignore[attr-defined]
    types_module.Content = FakeContent  # type: ignore[attr-defined]
    types_module.File = FakeFile  # type: ignore[attr-defined]
    types_module.FileState = SimpleNamespace(ACTIVE="ACTIVE")  # type: ignore[attr-defined]
    types_module.FileSource = SimpleNamespace(UPLOADED="UPLOADED")  # type: ignore[attr-defined]

    genai_module = ModuleType("google.genai")
    genai_module.types = types_module  # type: ignore[attr-defined]
    if client_factory is not None:
        genai_module.Client = client_factory  # type: ignore[attr-defined]

    google_module = ModuleType("google")
    google_module.genai = genai_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)


def test_image_to_genai_fetches_url_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")
    monkeypatch.setattr(
        multimodal.requests,
        "get",
        lambda url: SimpleNamespace(content=f"fetched:{url}".encode()),
    )

    image = Image(
        source="https://example.com/image.png",
        media_type="image/png",
        data=None,
    )

    part = multimodal.image_to_genai(image)

    assert part.data == b"fetched:https://example.com/image.png"
    assert part.mime_type == "image/png"


def test_image_to_genai_decodes_inline_data(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")

    image = Image(
        source="data:image/png;base64,QQ==",
        media_type="image/png",
        data="QQ==",
    )

    part = multimodal.image_to_genai(image)

    assert part.data == b"A"
    assert part.mime_type == "image/png"


def test_image_to_genai_raises_when_data_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai(monkeypatch)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")

    image = Image(source="local.png", media_type="image/png", data=None)

    with pytest.raises(ValueError, match="Image data is missing"):
        multimodal.image_to_genai(image)


def test_pdf_to_genai_handles_remote_and_inline_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai(monkeypatch)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")
    monkeypatch.setattr(
        multimodal.requests,
        "get",
        lambda _url: SimpleNamespace(content=b"remote-pdf"),
    )

    remote_pdf = PDF(
        source="https://example.com/file.pdf",
        media_type="application/pdf",
        data=None,
    )
    inline_pdf = PDF(
        source="file.pdf",
        media_type="application/pdf",
        data="QQ==",
    )

    remote_part = multimodal.pdf_to_genai(remote_pdf)
    inline_part = multimodal.pdf_to_genai(inline_pdf)

    assert remote_part.data == b"remote-pdf"
    assert inline_part.data == b"A"


def test_pdf_to_genai_raises_for_unsupported_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai(monkeypatch)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")

    with pytest.raises(ValueError, match="Unsupported PDF format"):
        multimodal.pdf_to_genai(
            PDF(source="file.pdf", media_type="application/pdf", data=None)
        )


def test_uploaded_pdf_to_genai_prefers_uri_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai(monkeypatch)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")

    uploaded = PDFWithGenaiFile(
        source="https://generativelanguage.googleapis.com/v1beta/files/123",
        media_type="application/pdf",
        data=None,
    )

    part = multimodal.uploaded_pdf_to_genai(uploaded)

    assert part.file_uri == uploaded.source
    assert part.mime_type == uploaded.media_type


def test_uploaded_pdf_to_genai_falls_back_to_pdf_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai(monkeypatch)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")
    sentinel = object()
    monkeypatch.setattr(multimodal, "pdf_to_genai", lambda _pdf: sentinel)

    pdf = PDFWithGenaiFile(
        source="file.pdf",
        media_type="application/pdf",
        data="QQ==",
    )

    assert multimodal.uploaded_pdf_to_genai(pdf) is sentinel


def test_upload_new_pdf_file_waits_until_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_states = iter(
        [
            SimpleNamespace(
                state="PENDING",
                name="files/1",
                uri="gs://file",
                mime_type="application/pdf",
            ),
            SimpleNamespace(
                state="ACTIVE",
                name="files/1",
                uri="gs://file",
                mime_type="application/pdf",
            ),
        ]
    )

    class FakeFiles:
        def upload(self, file: str) -> Any:
            assert file == "/tmp/demo.pdf"
            return next(file_states)

        def get(self, name: str) -> Any:
            assert name == "files/1"
            return next(file_states)

    class FakeClient:
        def __init__(self) -> None:
            self.files = FakeFiles()

    _install_fake_genai(monkeypatch, client_factory=FakeClient)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")
    sleeps: list[int] = []
    monkeypatch.setitem(sys.modules, "time", SimpleNamespace(sleep=sleeps.append))

    result = multimodal.upload_new_pdf_file(PDFWithGenaiFile, "/tmp/demo.pdf", 1, 2)

    assert sleeps == [1]
    assert result.source == "gs://file"
    assert result.media_type == "application/pdf"


def test_load_existing_pdf_file_accepts_uploaded_active_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFiles:
        def get(self, name: str) -> Any:
            assert name == "files/ready"
            return SimpleNamespace(
                source="UPLOADED",
                state="ACTIVE",
                uri="gs://ready",
                mime_type="application/pdf",
            )

    class FakeClient:
        def __init__(self) -> None:
            self.files = FakeFiles()

    _install_fake_genai(monkeypatch, client_factory=FakeClient)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")

    result = multimodal.load_existing_pdf_file(PDFWithGenaiFile, "files/ready")

    assert result.source == "gs://ready"
    assert result.media_type == "application/pdf"


def test_load_existing_pdf_file_rejects_non_uploaded_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFiles:
        def get(self, *, name: str) -> Any:
            assert name == "files/other"
            return SimpleNamespace(
                source="OTHER",
                state="ACTIVE",
                uri="gs://other",
                mime_type="application/pdf",
            )

    class FakeClient:
        def __init__(self) -> None:
            self.files = FakeFiles()

    _install_fake_genai(monkeypatch, client_factory=FakeClient)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")

    with pytest.raises(ValueError, match="uploaded PDFs"):
        multimodal.load_existing_pdf_file(PDFWithGenaiFile, "files/other")


def test_extract_multimodal_content_converts_detected_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai(monkeypatch)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")
    monkeypatch.setattr(Image, "to_genai", lambda _self: "converted-image")
    monkeypatch.setattr(
        multimodal,
        "autodetect_media",
        lambda text: Image(
            source="data:image/png;base64,QQ==",
            media_type="image/png",
            data="QQ==",
        )
        if text == "look at this"
        else text,
    )

    content = FakeContent(
        role="user",
        parts=[SimpleNamespace(text="look at this"), SimpleNamespace(text="leave me")],
    )

    result = multimodal.extract_multimodal_content([content], autodetect_images=True)

    assert len(result) == 1
    assert result[0].parts == ["converted-image", content.parts[1]]


def test_extract_multimodal_content_validates_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai(monkeypatch)
    multimodal = importlib.import_module("instructor.v2.providers.genai.multimodal")

    with pytest.raises(ValueError, match="Unsupported content type"):
        multimodal.extract_multimodal_content([object()])

    with pytest.raises(ValueError, match="Content parts are empty"):
        multimodal.extract_multimodal_content([FakeContent(role="user", parts=[])])
