"""Anthropic-specific multimodal encoders."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

import requests

from instructor.v2.core.multimodal import (
    Audio,
    Image,
    ImageParams,
    PDF,
)


class CacheableImage(Image):
    """Anthropic-owned image representation with optional prompt caching."""

    cache_control: Mapping[str, str] | None = None


def image_from_params(params: ImageParams) -> Image:
    """Construct an Anthropic image from its cache-aware shorthand."""
    image = Image.autodetect(params["source"])
    return CacheableImage(
        source=image.source,
        media_type=image.media_type,
        data=image.data,
        cache_control=params.get("cache_control"),
    )


def image_to_anthropic(image: Any) -> dict[str, Any]:
    if (
        isinstance(image.source, str)
        and image.source.startswith(("http://", "https://"))
        and not image.data
    ):
        image.data = image.url_to_base64(image.source)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": image.media_type,
            "data": image.data,
        },
    }


def pdf_to_anthropic(pdf: Any) -> dict[str, Any]:
    if (
        isinstance(pdf.source, str)
        and pdf.source.startswith(("http://", "https://"))
        and not pdf.data
    ):
        return {"type": "document", "source": {"type": "url", "url": pdf.source}}
    if not pdf.data:
        pdf.data = requests.get(str(pdf.source)).content
        pdf.data = base64.b64encode(pdf.data).decode("utf-8")
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": pdf.media_type,
            "data": pdf.data,
        },
    }


def image_with_cache_control_to_anthropic(image: Any) -> dict[str, Any]:
    result = image_to_anthropic(image)
    if cache_control := getattr(image, "cache_control", None):
        result["cache_control"] = cache_control
    return result


def pdf_with_cache_control_to_anthropic(pdf: Any) -> dict[str, Any]:
    result = pdf_to_anthropic(pdf)
    if cache_control := getattr(pdf, "cache_control", None):
        result["cache_control"] = cache_control
    return result


def audio_to_anthropic(_audio: Any) -> dict[str, Any]:
    raise NotImplementedError("Anthropic is not supported yet")


def media_to_anthropic(media: Image | Audio | PDF) -> dict[str, Any]:
    """Encode a typed media item through Anthropic-owned conversion."""
    if isinstance(media, Image):
        return image_with_cache_control_to_anthropic(media)
    if isinstance(media, PDF):
        return pdf_with_cache_control_to_anthropic(media)
    return audio_to_anthropic(media)
