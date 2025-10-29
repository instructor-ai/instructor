from __future__ import annotations
import base64
import pytest
from instructor.providers.bedrock.utils import (
    _openai_image_part_to_bedrock,
    _to_bedrock_content_items,
)


def test_openai_image_part_to_bedrock_data_url(tiny_png_data_url: str):
    part = {"type": "image_url", "image_url": {"url": tiny_png_data_url}}
    out = _openai_image_part_to_bedrock(part)
    assert "image" in out
    assert out["image"]["format"] in {"png", "jpeg", "gif", "webp"}  # png expected
    assert out["image"]["source"]["bytes"] == base64.b64decode(
        tiny_png_data_url.split(",", 1)[1]
    )


def test_openai_image_part_to_bedrock_https(image_url: str):
    part = {"type": "image_url", "image_url": {"url": image_url}}
    out = _openai_image_part_to_bedrock(part)
    assert "image" in out
    # GitHub raw returns jpeg for the sample. Normalize is handled in utils.
    assert out["image"]["format"] in {"jpeg", "png", "gif", "webp"}
    assert isinstance(out["image"]["source"]["bytes"], (bytes, bytearray))
    assert len(out["image"]["source"]["bytes"]) > 0


@pytest.mark.parametrize(
    "text_part",
    [
        {"type": "text", "text": "What is in this image?"},
        {"type": "input_text", "text": "Describe the image."},
    ],
)
@pytest.mark.parametrize("image_kind", ["data", "https"])
def test_to_bedrock_content_items_openai_combo(
    text_part, image_kind, tiny_png_data_url: str, image_url: str
):
    if image_kind == "data":
        image_part = {"type": "image_url", "image_url": {"url": tiny_png_data_url}}
    else:
        image_part = {"type": "image_url", "image_url": {"url": image_url}}

    content = [text_part, image_part]
    items = _to_bedrock_content_items(content)

    assert items[0] == {"text": text_part["text"]}
    assert "image" in items[1]
    assert isinstance(items[1]["image"]["source"]["bytes"], (bytes, bytearray))
    assert len(items[1]["image"]["source"]["bytes"]) > 0
