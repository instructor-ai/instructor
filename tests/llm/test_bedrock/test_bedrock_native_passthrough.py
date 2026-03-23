from __future__ import annotations
from instructor.providers.bedrock.utils import _to_bedrock_content_items


def test_bedrock_native_text_passthrough():
    content = [{"text": "Bedrock-native text"}]
    items = _to_bedrock_content_items(content)
    assert items == [{"text": "Bedrock-native text"}]


def test_bedrock_native_image_passthrough(tiny_png_bytes: bytes):
    native = {"image": {"format": "png", "source": {"bytes": tiny_png_bytes}}}
    items = _to_bedrock_content_items([native])
    assert items[0] == native


def test_bedrock_native_document_passthrough(tiny_pdf_bytes: bytes):
    native = {"document": {"format": "pdf", "source": {"bytes": tiny_pdf_bytes}}}
    items = _to_bedrock_content_items([native])
    assert items[0] == native


def test_bedrock_native_cachepoint_passthrough():
    native = {"cachePoint": {"type": "default"}}
    items = _to_bedrock_content_items([native])
    assert items[0] == native


def test_bedrock_cachepoint_mixed_with_text():
    content = [
        {"text": "Hello world"},
        {"cachePoint": {"type": "default"}},
        {"text": "More text"},
    ]
    items = _to_bedrock_content_items(content)
    assert len(items) == 3
    assert items[0] == {"text": "Hello world"}
    assert items[1] == {"cachePoint": {"type": "default"}}
    assert items[2] == {"text": "More text"}
