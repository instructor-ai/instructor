"""Google GenAI-specific multimodal encoders."""

from __future__ import annotations

import base64
from typing import Any

import requests


def _types() -> Any:
    try:
        from google.genai import types
    except ImportError as err:
        raise ImportError(
            "google-genai package is required for GenAI integration. Install with: pip install google-genai"
        ) from err
    return types


def image_to_genai(image: Any) -> Any:
    types = _types()
    if isinstance(image.source, str) and image.source.startswith("gs://"):
        return types.Part.from_bytes(data=image.data, mime_type=image.media_type)
    if isinstance(image.source, str) and image.source.startswith(("http://", "https://")):
        return types.Part.from_bytes(
            data=requests.get(image.source).content,
            mime_type=image.media_type,
        )
    if image.data or image.is_base64(str(image.source)):
        data = image.data or str(image.source).split(",", 1)[1]
        return types.Part.from_bytes(
            data=base64.b64decode(data),
            mime_type=image.media_type,
        )
    raise ValueError("Image data is missing for base64 encoding.")


def audio_to_genai(audio: Any) -> Any:
    types = _types()
    return types.Part.from_bytes(
        data=base64.b64decode(audio.data),
        mime_type=audio.media_type,
    )


def pdf_to_genai(pdf: Any) -> Any:
    types = _types()
    if (
        isinstance(pdf.source, str)
        and pdf.source.startswith(("http://", "https://"))
        and not pdf.data
    ):
        data = requests.get(pdf.source).content
        encoded = base64.b64encode(data).decode("utf-8")
        return types.Part.from_bytes(
            data=base64.b64decode(encoded),
            mime_type=pdf.media_type,
        )
    if pdf.data:
        return types.Part.from_bytes(
            data=base64.b64decode(pdf.data),
            mime_type=pdf.media_type,
        )
    raise ValueError("Unsupported PDF format")
