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
