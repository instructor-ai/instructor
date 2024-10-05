from __future__ import annotations
import base64
from typing import Any
from pathlib import Path
from pydantic import BaseModel, Field
from .mode import Mode


class Image(BaseModel):
    """Represents an image that can be loaded from a URL or file path."""

    source: str | Path = Field(..., description="URL or file path of the image")
    media_type: str = Field(..., description="MIME type of the image")
    data: str | None = Field(None, description="Base64 encoded image data", repr=False)

    @classmethod
    def from_url(cls, url: str) -> Image:
        """Create an Image instance from a URL."""
        return cls(source=url, media_type="image/jpeg", data=None)

    @classmethod
    def from_path(cls, path: str | Path) -> Image:
        """Create an Image instance from a file path."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        suffix = path.suffix.lower().lstrip(".")
        if suffix not in ["jpeg", "jpg", "png"]:
            raise ValueError(f"Unsupported image format: {suffix}")

        media_type = "image/jpeg" if suffix in ["jpeg", "jpg"] else "image/png"
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(source=str(path), media_type=media_type, data=data)

    def to_anthropic(self) -> dict[str, Any]:
        """Convert the Image instance to Anthropic's API format."""
        if isinstance(self.source, str) and self.source.startswith(
            ("http://", "https://")
        ):
            import requests

            response = requests.get(self.source)
            response.raise_for_status()
            self.data = base64.b64encode(response.content).decode("utf-8")
            self.media_type = response.headers.get("Content-Type", "image/jpeg")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.data,
            },
        }

    def to_openai(self) -> dict[str, Any]:
        """Convert the Image instance to OpenAI's Vision API format."""
        if isinstance(self.source, str) and self.source.startswith(
            ("http://", "https://")
        ):
            return {"type": "image_url", "image_url": {"url": self.source}}
        elif self.data:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{self.media_type};base64,{self.data}"},
            }
        else:
            raise ValueError("Image data is missing for base64 encoding.")


def convert_contents(
    contents: list[str | Image] | str | Image, mode: Mode
) -> str | list[dict[str, Any]]:
    """Convert content items to the appropriate format based on the specified mode."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, Image):
        contents = [contents]

    converted_contents: list[dict[str, str | Image]] = []
    for content in contents:
        if isinstance(content, str):
            converted_contents.append({"type": "text", "text": content})
        elif isinstance(content, Image):
            if mode in {Mode.ANTHROPIC_JSON, Mode.ANTHROPIC_TOOLS}:
                converted_contents.append(content.to_anthropic())
            elif mode in {Mode.GEMINI_JSON, Mode.GEMINI_TOOLS}:
                raise NotImplementedError("Gemini is not supported yet")
            else:
                converted_contents.append(content.to_openai())
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    return converted_contents


def convert_messages(
    messages: list[dict[str, str | list[str | Image]]], mode: Mode
) -> list[dict[str, Any]]:
    """Convert messages to the appropriate format based on the specified mode."""
    converted_messages = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if isinstance(content, str):
            converted_messages.append({"role": role, "content": content})  # type: ignore
        else:
            converted_content = convert_contents(content, mode)
            converted_messages.append({"role": role, "content": converted_content})  # type: ignore
    return converted_messages  # type: ignore
