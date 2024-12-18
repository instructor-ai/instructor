from __future__ import annotations
from .mode import Mode
import base64
import re
from collections.abc import Mapping, Hashable
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    TypedDict,
    TypeVar,
    cast,
)
from pathlib import Path
from urllib.parse import urlparse
import mimetypes
import requests
from pydantic import BaseModel, Field
from .mode import Mode

F = TypeVar("F", bound=Callable[..., Any])
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

# OpenAI source: https://platform.openai.com/docs/guides/vision/what-type-of-files-can-i-upload
# Anthropic source: https://docs.anthropic.com/en/docs/build-with-claude/vision#ensuring-image-quality
VALID_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
CacheControlType = Mapping[str, str]
OptionalCacheControlType = Optional[CacheControlType]


class ImageParamsBase(TypedDict):
    type: Literal["image"]
    source: str


class ImageParams(ImageParamsBase, total=False):
    cache_control: CacheControlType


class Image(BaseModel):
    source: Union[str, Path] = Field(  # noqa: UP007
        description="URL, file path, or base64 data of the image"
    )
    media_type: str = Field(description="MIME type of the image")
    data: Union[str, None] = Field(  # noqa: UP007
        None, description="Base64 encoded image data", repr=False
    )

    @classmethod
    def autodetect(cls, source: Union[str, Path]) -> Image:  # noqa: UP007
        """Attempt to autodetect an image from a source string or Path.

        Args:
            source (Union[str,path]): The source string or path.
        Returns:
            An Image if the source is detected to be a valid image.
        Raises:
            ValueError: If the source is not detected to be a valid image.
        """
        if isinstance(source, str):
            if cls.is_base64(source):
                return cls.from_base64(source)
            elif source.startswith(("http://", "https://")):
                return cls.from_url(source)
            elif Path(source).is_file():
                return cls.from_path(source)
            else:
                return cls.from_raw_base64(source)
        elif isinstance(source, Path):
            return cls.from_path(source)

        raise ValueError("Unable to determine image type or unsupported image format")

    @classmethod
    def autodetect_safely(cls, source: str | Path) -> Union[Image, str]:  # noqa: UP007
        """Safely attempt to autodetect an image from a source string or path.

        Args:
            source (Union[str,path]): The source string or path.
        Returns:
            An Image if the source is detected to be a valid image, otherwise
            the source itself as a string.
        """
        try:
            return cls.autodetect(source)
        except ValueError:
            return str(source)

    @classmethod
    def is_base64(cls, s: str) -> bool:
        return bool(re.match(r"^data:image/[a-zA-Z]+;base64,", s))

    @classmethod  # Caching likely unnecessary
    def from_base64(cls, data_uri: str) -> Image:
        header, encoded = data_uri.split(",", 1)
        media_type = header.split(":")[1].split(";")[0]
        if media_type not in VALID_MIME_TYPES:
            raise ValueError(f"Unsupported image format: {media_type}")
        return cls(
            source=data_uri,
            media_type=media_type,
            data=encoded,
        )

    @classmethod  # Caching likely unnecessary
    def from_raw_base64(cls, data: str) -> Image:
        try:
            decoded = base64.b64decode(data)
            import imghdr

            img_type = imghdr.what(None, decoded)
            if img_type:
                media_type = f"image/{img_type}"
                if media_type in VALID_MIME_TYPES:
                    return cls(
                        source=data,
                        media_type=media_type,
                        data=data,
                    )
            raise ValueError(f"Unsupported image type: {img_type}")
        except Exception as e:
            raise ValueError(f"Invalid or unsupported base64 image data") from e

    @classmethod
    @lru_cache
    def from_url(cls, url: str) -> Image:
        if cls.is_base64(url):
            return cls.from_base64(url)

        parsed_url = urlparse(url)
        media_type, _ = mimetypes.guess_type(parsed_url.path)

        if not media_type:
            try:
                response = requests.head(url, allow_redirects=True)
                media_type = response.headers.get("Content-Type")
            except requests.RequestException as e:
                raise ValueError(f"Failed to fetch image from URL") from e

        if media_type not in VALID_MIME_TYPES:
            raise ValueError(f"Unsupported image format: {media_type}")
        return cls(source=url, media_type=media_type, data=None)

    @classmethod
    @lru_cache
    def from_path(cls, path: Union[str, Path]) -> Image:  # noqa: UP007
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError("Image file is empty")

        media_type, _ = mimetypes.guess_type(str(path))
        if media_type not in VALID_MIME_TYPES:
            raise ValueError(f"Unsupported image format: {media_type}")

        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(source=path, media_type=media_type, data=data)

    @staticmethod
    @lru_cache
    def url_to_base64(url: str) -> str:
        """Cachable helper method for getting image url and encoding to base64."""
        response = requests.get(url)
        response.raise_for_status()
        data = base64.b64encode(response.content).decode("utf-8")
        return data

    def to_anthropic(self) -> dict[str, Any]:
        if (
            isinstance(self.source, str)
            and self.source.startswith(("http://", "https://"))
            and not self.data
        ):
            self.data = self.url_to_base64(self.source)

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.data,
            },
        }

    def to_openai(self) -> dict[str, Any]:
        if (
            isinstance(self.source, str)
            and self.source.startswith(("http://", "https://"))
            and not self.is_base64(self.source)
        ):
            return {"type": "image_url", "image_url": {"url": self.source}}
        elif self.data or self.is_base64(str(self.source)):
            data = self.data or str(self.source).split(",", 1)[1]
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{self.media_type};base64,{data}"},
            }
        else:
            raise ValueError("Image data is missing for base64 encoding.")


class Audio(BaseModel):
    """Represents an audio that can be loaded from a URL or file path."""

    source: str | Path = Field(description="URL or file path of the audio")  # noqa: UP007
    data: Union[str, None] = Field(  # noqa: UP007
        None, description="Base64 encoded audio data", repr=False
    )

    @classmethod
    def from_url(cls, url: str) -> Audio:
        """Create an Audio instance from a URL."""
        assert url.endswith(".wav"), "Audio must be in WAV format"

        response = requests.get(url)
        data = base64.b64encode(response.content).decode("utf-8")
        return cls(source=url, data=data)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> Audio:  # noqa: UP007
        """Create an Audio instance from a file path."""
        path = Path(path)
        assert path.is_file(), f"Audio file not found: {path}"
        assert path.suffix.lower() == ".wav", "Audio must be in WAV format"

        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(source=str(path), data=data)

    def to_openai(self) -> dict[str, Any]:
        """Convert the Audio instance to OpenAI's API format."""
        return {
            "type": "input_audio",
            "input_audio": {"data": self.data, "format": "wav"},
        }

    def to_anthropic(self) -> dict[str, Any]:
        raise NotImplementedError("Anthropic is not supported yet")


class ImageWithCacheControl(Image):
    """Image with Anthropic prompt caching support."""

    cache_control: OptionalCacheControlType = Field(
        None, description="Optional Anthropic cache control image"
    )

    @classmethod
    def from_image_params(cls, image_params: ImageParams) -> Image:
        source = image_params["source"]
        cache_control = image_params.get("cache_control")
        base_image = Image.autodetect(source)
        return cls(
            source=base_image.source,
            media_type=base_image.media_type,
            data=base_image.data,
            cache_control=cache_control,
        )

    def to_anthropic(self) -> dict[str, Any]:
        """Override Anthropic return with cache_control."""
        result = super().to_anthropic()
        if self.cache_control:
            result["cache_control"] = self.cache_control
        return result


def convert_contents(
    contents: Union[  # noqa: UP007
        str,
        dict[str, Any],
        Image,
        Audio,
        list[Union[str, dict[str, Any], Image, Audio]],  # noqa: UP007
    ],
    mode: Mode,
) -> Union[str, list[dict[str, Any]]]:  # noqa: UP007
    """Convert content items to the appropriate format based on the specified mode."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, (Image, Audio)) or isinstance(contents, dict):
        contents = [contents]

    converted_contents: list[dict[str, Union[str, Image]]] = []  # noqa: UP007
    for content in contents:
        if isinstance(content, str):
            converted_contents.append({"type": "text", "text": content})
        elif isinstance(content, dict):
            converted_contents.append(content)
        elif isinstance(content, (Image, Audio)):
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
    messages: list[
        dict[
            str,
            Union[  # noqa: UP007
                str,
                dict[str, Any],
                Image,
                Audio,
                list[Union[str, dict[str, Any], Image, Audio]],  # noqa: UP007
            ],
        ]
    ],
    mode: Mode,
    autodetect_images: bool = False,
) -> list[dict[str, Any]]:
    """Convert messages to the appropriate format based on the specified mode."""
    converted_messages = []

    def is_image_params(x: Any) -> bool:
        return isinstance(x, dict) and x.get("type") == "image" and "source" in x  # type: ignore

    for message in messages:
        if "type" in message:
            if message["type"] in {"audio", "image"}:
                converted_messages.append(message)  # type: ignore
            else:
                raise ValueError(f"Unsupported message type: {message['type']}")
        role = message["role"]
        content = message["content"] or []
        other_kwargs = {
            k: v for k, v in message.items() if k not in ["role", "content", "type"]
        }
        if autodetect_images:
            if isinstance(content, list):
                new_content: list[str | dict[str, Any] | Image | Audio] = []  # noqa: UP007
                for item in content:
                    if isinstance(item, str):
                        new_content.append(Image.autodetect_safely(item))
                    elif is_image_params(item):
                        new_content.append(
                            ImageWithCacheControl.from_image_params(
                                cast(ImageParams, item)
                            )
                        )
                    else:
                        new_content.append(item)
                content = new_content
            elif isinstance(content, str):
                content = Image.autodetect_safely(content)
            elif is_image_params(content):
                content = ImageWithCacheControl.from_image_params(
                    cast(ImageParams, content)
                )
        if isinstance(content, str):
            converted_messages.append(  # type: ignore
                {"role": role, "content": content, **other_kwargs}
            )
        else:
            converted_content = convert_contents(content, mode)
            converted_messages.append(  # type: ignore
                {"role": role, "content": converted_content, **other_kwargs}
            )
    return converted_messages  # type: ignore
