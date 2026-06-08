"""OpenAI-specific multimodal encoders."""

from __future__ import annotations

import base64
from typing import Any

import requests

from instructor.v2.core.mode import Mode
from instructor.v2.core.multimodal import Audio, Image, ImageParams, PDF

RESPONSES_MODES = {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}


def image_from_params(params: ImageParams) -> Image:
    """Construct an OpenAI image from the provider-neutral image shorthand."""
    return Image.autodetect(params["source"])


def image_to_openai(image: Any, mode: Mode) -> dict[str, Any]:
    image_type = "input_image" if mode in RESPONSES_MODES else "image_url"
    if (
        isinstance(image.source, str)
        and image.source.startswith(("http://", "https://"))
        and not image.is_base64(image.source)
    ):
        if mode in RESPONSES_MODES:
            return {"type": "input_image", "image_url": image.source}
        return {"type": image_type, "image_url": {"url": image.source}}
    if image.data or image.is_base64(str(image.source)):
        data = image.data or str(image.source).split(",", 1)[1]
        if mode in RESPONSES_MODES:
            return {
                "type": "input_image",
                "image_url": f"data:{image.media_type};base64,{data}",
            }
        return {
            "type": image_type,
            "image_url": {"url": f"data:{image.media_type};base64,{data}"},
        }
    raise ValueError("Image data is missing for base64 encoding.")


def audio_to_openai(audio: Any, mode: Mode) -> dict[str, Any]:
    if mode in RESPONSES_MODES:
        raise ValueError("OpenAI Responses doesn't support audio")
    return {"type": "input_audio", "input_audio": {"data": audio.data, "format": "wav"}}


def pdf_to_openai(pdf: Any, mode: Mode) -> dict[str, Any]:
    input_file_type = "input_file" if mode in RESPONSES_MODES else "file"
    if (
        isinstance(pdf.source, str)
        and pdf.source.startswith(("http://", "https://"))
        and not pdf.data
    ):
        response = requests.get(pdf.source)
        data = base64.b64encode(response.content).decode("utf-8")
        if mode in RESPONSES_MODES:
            return {
                "type": input_file_type,
                "filename": pdf.source,
                "file_data": f"data:{pdf.media_type};base64,{data}",
            }
        return {
            "type": input_file_type,
            "file": {
                "filename": pdf.source,
                "file_data": f"data:{pdf.media_type};base64,{data}",
            },
        }
    if pdf.data or pdf.is_base64(str(pdf.source)):
        data = pdf.data or str(pdf.source).split(",", 1)[1]
        filename = pdf.source if isinstance(pdf.source, str) else str(pdf.source)
        if mode in RESPONSES_MODES:
            return {
                "type": input_file_type,
                "filename": filename,
                "file_data": f"data:{pdf.media_type};base64,{data}",
            }
        return {
            "type": input_file_type,
            "file": {
                "filename": filename,
                "file_data": f"data:{pdf.media_type};base64,{data}",
            },
        }
    raise ValueError("PDF data is missing for base64 encoding.")


def media_to_openai(media: Image | Audio | PDF, mode: Mode) -> dict[str, Any]:
    """Encode a typed media item through OpenAI-owned conversion."""
    if isinstance(media, Image):
        return image_to_openai(media, mode)
    if isinstance(media, Audio):
        return audio_to_openai(media, mode)
    return pdf_to_openai(media, mode)
