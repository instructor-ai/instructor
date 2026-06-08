"""Mistral-specific multimodal encoders."""

from __future__ import annotations

from typing import Any

from instructor.v2.core.mode import Mode
from instructor.v2.core.multimodal import Audio, Image, ImageParams, PDF
from instructor.v2.providers.openai.multimodal import audio_to_openai, image_to_openai


def image_from_params(params: ImageParams) -> Image:
    """Construct a Mistral image from the provider-neutral image shorthand."""
    return Image.autodetect(params["source"])


def pdf_to_mistral(pdf: Any) -> dict[str, Any]:
    if (
        isinstance(pdf.source, str)
        and pdf.source.startswith(("http://", "https://"))
        and not pdf.data
    ):
        return {"type": "document_url", "document_url": pdf.source}
    raise ValueError("Mistral only supports document URLs for now")


def media_to_mistral(media: Image | Audio | PDF, mode: Mode) -> dict[str, Any]:
    """Encode media through the Mistral-owned OpenAI-compatible boundary."""
    if isinstance(media, PDF):
        return pdf_to_mistral(media)
    if isinstance(media, Image):
        return image_to_openai(media, mode)
    return audio_to_openai(media, mode)
