from __future__ import annotations
from .mode import Mode  # Required for image format conversion
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
from pydantic import BaseModel
from pydantic.fields import Field

# Constants for Mistral image validation
VALID_MISTRAL_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_MISTRAL_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB in bytes

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
    source: Union[str, Path] = Field(
        description="URL, file path, or base64 data of the image"
    )
    media_type: str = Field(description="MIME type of the image")
    data: Union[str, None] = Field(
        None, description="Base64 encoded image data", repr=False
    )
    @classmethod
    def autodetect(cls, source: Union[str, Path]) -> "Image":
        """Attempt to autodetect an image from a source string or Path.

        Args:
            source (Union[str, Path]): The source string or path.
        Returns:
            Image: An Image if the source is detected to be a valid image.
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
    def autodetect_safely(
        cls, source: Union[str, Path]
    ) -> Union["Image", str]:
        """Safely attempt to autodetect an image from a source string or path.

        Args:
            source (Union[str, Path]): The source string or path.
        Returns:
            Union[Image, str]: An Image if the source is detected to be a valid image, otherwise
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
    def from_base64(cls, data_uri: str) -> "Image":
        header: str
        encoded: str
        header, encoded = data_uri.split(",", 1)
        media_type: str = header.split(":")[1].split(";")[0]
        if media_type not in VALID_MIME_TYPES:
            raise ValueError(f"Unsupported image format: {media_type}")
        return cls(
            source=data_uri,
            media_type=media_type,
            data=encoded,
        )

    @classmethod  # Caching likely unnecessary
    def from_raw_base64(cls, data: str) -> "Image":
        try:
            decoded: bytes = base64.b64decode(data)
            import imghdr

            img_type: Union[str, None] = imghdr.what(None, decoded)
            if img_type:
                media_type: str = f"image/{img_type}"
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
    def from_url(cls, url: str) -> "Image":
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
    def from_path(cls, path: Union[str, Path]) -> "Image":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError("Image file is empty")

        if path.stat().st_size > MAX_MISTRAL_IMAGE_SIZE:
            raise ValueError(f"Image file size ({path.stat().st_size / 1024 / 1024:.1f}MB) "
                           f"exceeds Mistral's limit of {MAX_MISTRAL_IMAGE_SIZE / 1024 / 1024:.1f}MB")

        media_type, _ = mimetypes.guess_type(str(path))
        if media_type not in VALID_MISTRAL_MIME_TYPES:
            raise ValueError(f"Unsupported image format: {media_type}. "
                           f"Supported formats are: {', '.join(VALID_MISTRAL_MIME_TYPES)}")

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

    def to_mistral(self) -> dict[str, Any]:
        """Convert the image to Mistral's API format.

        Returns:
            dict[str, Any]: Image data in Mistral's API format, either as a URL or base64 data URI.

        Raises:
            ValueError: If the image format is not supported by Mistral or exceeds size limit.
        """
        # Validate media type
        if self.media_type not in VALID_MISTRAL_MIME_TYPES:
            raise ValueError(f"Unsupported image format for Mistral: {self.media_type}. "
                           f"Supported formats are: {', '.join(VALID_MISTRAL_MIME_TYPES)}")

        # For base64 data, validate size
        if self.data:
            # Calculate size of decoded base64 data
            data_size = len(base64.b64decode(self.data))
            if data_size > MAX_MISTRAL_IMAGE_SIZE:
                raise ValueError(f"Image size ({data_size / 1024 / 1024:.1f}MB) exceeds "
                               f"Mistral's limit of {MAX_MISTRAL_IMAGE_SIZE / 1024 / 1024:.1f}MB")

        if (
            isinstance(self.source, str)
            and self.source.startswith(("http://", "https://"))
            and not self.is_base64(self.source)
        ):
            return {"type": "image_url", "url": self.source}
        elif self.data or self.is_base64(str(self.source)):
            data = self.data or str(self.source).split(",", 1)[1]
            return {
                "type": "image_url",
                "data": f"data:{self.media_type};base64,{data}"
            }
        else:
            raise ValueError("Image data is missing for base64 encoding.")

    """Represents an audio that can be loaded from a URL or file path."""
    source: Union[str, Path] = Field(
        description="URL or file path of the audio"
    )
    data: Union[str, None] = Field(
        None, description="Base64 encoded audio data", repr=False
    )

    # PLACEHOLDER: Image class methods and properties above

    # PLACEHOLDER: ImageWithCacheControl class below


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
    contents: list[Union[str, Image]], mode: Mode
) -> list[Union[str, dict[str, Any]]]:
    """Convert contents to the appropriate format for the given mode."""
    converted_contents: list[Union[str, dict[str, Any]]] = []
    for content in contents:
        if isinstance(content, str):
            converted_contents.append(content)
        elif isinstance(content, Image):
            if mode in {Mode.ANTHROPIC_JSON, Mode.ANTHROPIC_TOOLS}:
                converted_contents.append(content.to_anthropic())
            elif mode in {Mode.GEMINI_JSON, Mode.GEMINI_TOOLS}:
                raise NotImplementedError("Gemini is not supported yet")
            elif mode in {Mode.MISTRAL_JSON, Mode.MISTRAL_TOOLS}:
                converted_contents.append(content.to_mistral())
            else:
                converted_contents.append(content.to_openai())
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    return converted_contents


def convert_messages(
    messages: list[dict[str, Any]],
    mode: Mode,
) -> list[dict[str, Any]]:
    """Convert messages to the appropriate format for the given mode.

    Args:
        messages: List of message dictionaries to convert
        mode: The mode to convert messages for (e.g. MISTRAL_JSON)

    Returns:
        List of converted message dictionaries
    """
    if mode == Mode.MISTRAL_JSON:
        converted_messages: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message.get("content"), list):
                converted_messages.append(message)
                continue

            content_list: list[dict[str, Any]] = []
            for item in cast(list[Union[str, Image, dict[str, Any]]], message["content"]):
                if isinstance(item, str):
                    content_list.append({"type": "text", "text": item})
                elif isinstance(item, Image):
                    content_list.append(item.to_mistral())
                else:
                    content_list.append(item)  # item is already dict[str, Any]

            converted_message = message.copy()
            converted_message["content"] = content_list
            converted_messages.append(converted_message)

        return converted_messages

    # Return original messages for other modes
    return messages
