from __future__ import annotations

import pytest

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
    """Regression test for #1954: cachePoint dicts must pass through."""
    cache_point = {"cachePoint": {"type": "default"}}
    items = _to_bedrock_content_items([cache_point])
    assert items == [cache_point]


def test_bedrock_native_cachepoint_with_ttl():
    cache_point = {"cachePoint": {"type": "default", "ttl": "5m"}}
    items = _to_bedrock_content_items([cache_point])
    assert items == [cache_point]


def test_bedrock_native_guard_content_passthrough():
    guard = {"guardContent": {"text": {"text": "test content"}}}
    items = _to_bedrock_content_items([guard])
    assert items == [guard]


@pytest.mark.parametrize(
    "block",
    [
        {"cachePoint": {"type": "default"}},
        {"guardContent": {"text": {"text": "check this"}}},
        {"video": {"format": "mp4", "source": {"bytes": b"fake"}}},
        {"audio": {"format": "mp3", "source": {"bytes": b"fake"}}},
    ],
    ids=["cachePoint", "guardContent", "video", "audio"],
)
def test_bedrock_native_content_block_passthrough(block: dict):
    """All Bedrock-native content blocks should pass through unchanged."""
    items = _to_bedrock_content_items([block])
    assert items == [block]


def test_mixed_content_with_cachepoint():
    """Regression test for #1954: cachePoint mixed with text in real usage."""
    content = [
        {"text": "Say hello world."},
        {"cachePoint": {"type": "default"}},
        {"text": "This is a test message."},
    ]
    items = _to_bedrock_content_items(content)
    assert items == content
