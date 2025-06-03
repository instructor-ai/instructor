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
VALID_AUDIO_MIME_TYPES = [
    "audio/aac",
    "audio/flac",
    "audio/mp3",
    "audio/m4a",
    "audio/mpeg",
    "audio/mpga",
    "audio/mp4",
    "audio/opus",
    "audio/pcm",
    "audio/wav",
    "audio/webm",
]
VALID_PDF_MIME_TYPES = ["application/pdf"]
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
    def autodetect(cls, source: str | Path) -> Image:
        """Attempt to autodetect an image from a source string or Path."""
        if isinstance(source, str):
            if cls.is_base64(source):
                return cls.from_base64(source)
            if source.startswith(("http://", "https://")):
                return cls.from_url(source)
            if source.startswith("gs://"):
                return cls.from_gs_url(source)
            # Since detecting the max length of a file universally cross-platform is difficult,
            # we'll just try/catch the Path conversion and file check
            try:
                path = Path(source)
                if path.is_file():
                    return cls.from_path(path)
            except OSError:
                pass  # Fall through to raw base64 attempt

            return cls.from_raw_base64(source)

        if isinstance(source, Path):
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

    @classmethod
    def from_gs_url(cls, data_uri: str) -> Image:
        """
        Create an Image instance from a Google Cloud Storage URL.
        """
        if not data_uri.startswith("gs://"):
            raise ValueError("URL must start with gs://")

        public_url = f"https://storage.googleapis.com/{data_uri[5:]}"

        try:
            response = requests.get(public_url)
            response.raise_for_status()
            media_type = response.headers.get("Content-Type")
            if media_type not in VALID_MIME_TYPES:
                raise ValueError(f"Unsupported image format: {media_type}")

            data = base64.b64encode(response.content).decode("utf-8")

            return cls(source=data_uri, media_type=media_type, data=data)
        except requests.RequestException as e:
            raise ValueError(f"We only support public images for now") from e

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

    def to_openai(self, mode: Mode) -> dict[str, Any]:
        image_type = (
            "input_image"
            if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}
            else "image_url"
        )
        if (
            isinstance(self.source, str)
            and self.source.startswith(("http://", "https://"))
            and not self.is_base64(self.source)
        ):
            if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}:
                return {"type": "input_image", "image_url": self.source}
            else:
                return {"type": image_type, "image_url": {"url": self.source}}
        elif self.data or self.is_base64(str(self.source)):
            data = self.data or str(self.source).split(",", 1)[1]
            if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}:
                return {
                    "type": "input_image",
                    "image_url": f"data:{self.media_type};base64,{data}",
                }
            else:
                return {
                    "type": image_type,
                    "image_url": {"url": f"data:{self.media_type};base64,{data}"},
                }
        else:
            raise ValueError("Image data is missing for base64 encoding.")

    def to_genai(self):
        """
        Convert the Image instance to Google GenAI's API format.
        """
        from google.genai import types

        # Google Cloud Storage
        if isinstance(self.source, str) and self.source.startswith("gs://"):
            return types.Part.from_bytes(
                data=self.data,  # type: ignore
                mime_type=self.media_type,
            )

        # URL
        if isinstance(self.source, str) and self.source.startswith(
            ("http://", "https://")
        ):
            return types.Part.from_bytes(
                data=requests.get(self.source).content,
                mime_type=self.media_type,
            )

        if self.data or self.is_base64(str(self.source)):
            data = self.data or str(self.source).split(",", 1)[1]
            return types.Part.from_bytes(
                data=base64.b64decode(data), mime_type=self.media_type
            )  # type: ignore

        else:
            raise ValueError("Image data is missing for base64 encoding.")


class Audio(BaseModel):
    """Represents an audio that can be loaded from a URL or file path."""

    source: str | Path = Field(description="URL or file path of the audio")  # noqa: UP007
    data: Union[str, None] = Field(  # noqa: UP007
        None, description="Base64 encoded audio data", repr=False
    )
    media_type: str = Field(description="MIME type of the audio")

    @classmethod
    def from_url(cls, url: str) -> Audio:
        """Create an Audio instance from a URL."""
        response = requests.get(url)
        content_type = response.headers.get("content-type")
        assert content_type in VALID_AUDIO_MIME_TYPES, (
            f"Invalid audio format. Must be one of: {', '.join(VALID_AUDIO_MIME_TYPES)}"
        )

        data = base64.b64encode(response.content).decode("utf-8")
        return cls(source=url, data=data, media_type=content_type)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> Audio:  # noqa: UP007
        """Create an Audio instance from a file path."""
        path = Path(path)
        assert path.is_file(), f"Audio file not found: {path}"

        mime_type = mimetypes.guess_type(str(path))[0]

        if mime_type == "audio/x-wav":
            mime_type = "audio/wav"

        assert mime_type in VALID_AUDIO_MIME_TYPES, (
            f"Invalid audio format. Must be one of: {', '.join(VALID_AUDIO_MIME_TYPES)}"
        )

        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(source=str(path), data=data, media_type=mime_type)

    def to_openai(self, mode: Mode) -> dict[str, Any]:
        """Convert the Audio instance to OpenAI's API format."""
        if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}:
            raise ValueError("OpenAI Responses doesn't support audio")

        return {
            "type": "input_audio",
            "input_audio": {"data": self.data, "format": "wav"},
        }

    def to_anthropic(self) -> dict[str, Any]:
        raise NotImplementedError("Anthropic is not supported yet")

    def to_genai(self):
        """
        Convert the Audio instance to Google GenAI's API format.
        """
        from google.genai import types

        return types.Part.from_bytes(
            data=base64.b64decode(self.data),  # type: ignore
            mime_type=self.media_type,
        )


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


class PDF(BaseModel):
    source: str | Path = Field(description="URL, file path, or base64 data of the PDF")
    media_type: str = Field(
        description="MIME type of the PDF", default="application/pdf"
    )
    data: str | None = Field(None, description="Base64 encoded PDF data", repr=False)

    @classmethod
    def autodetect(cls, source: str | Path) -> PDF:
        """Attempt to autodetect a PDF from a source string or Path.
        Args:
            source (Union[str,path]): The source string or path.
        Returns:
            A PDF if the source is detected to be a valid PDF.
        Raises:
            ValueError: If the source is not detected to be a valid PDF.
        """
        if isinstance(source, str):
            if cls.is_base64(source):
                return cls.from_base64(source)
            elif source.startswith(("http://", "https://")):
                return cls.from_url(source)

            try:
                if Path(source).is_file():
                    return cls.from_path(source)
            except FileNotFoundError as err:
                raise ValueError("PDF file not found") from err
            except OSError as e:
                if e.errno == 63:  # File name too long
                    raise ValueError("PDF file name too long") from e
                raise ValueError("Unable to read PDF file") from e

            return cls.from_raw_base64(source)
        elif isinstance(source, Path):
            return cls.from_path(source)

        raise ValueError("Unable to determine PDF type or unsupported PDF format")

    @classmethod
    def is_base64(cls, s: str) -> bool:
        return bool(re.match(r"^data:application/pdf;base64,", s))

    @classmethod
    def from_base64(cls, data_uri: str) -> PDF:
        header, encoded = data_uri.split(",", 1)
        media_type = header.split(":")[1].split(";")[0]
        if media_type not in VALID_PDF_MIME_TYPES:
            raise ValueError(f"Unsupported PDF format: {media_type}")
        return cls(
            source=data_uri,
            media_type=media_type,
            data=encoded,
        )

    @classmethod
    @lru_cache
    def from_path(cls, path: str | Path) -> PDF:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError("PDF file is empty")

        media_type, _ = mimetypes.guess_type(str(path))
        if media_type not in VALID_PDF_MIME_TYPES:
            raise ValueError(f"Unsupported PDF format: {media_type}")

        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(source=path, media_type=media_type, data=data)

    @classmethod
    def from_raw_base64(cls, data: str) -> PDF:
        try:
            decoded = base64.b64decode(data)
            # Check if it's a valid PDF by looking for the PDF header
            if decoded.startswith(b"%PDF-"):
                return cls(
                    source=data,
                    media_type="application/pdf",
                    data=data,
                )
            raise ValueError("Invalid PDF format")
        except Exception as e:
            raise ValueError("Invalid or unsupported base64 PDF data") from e

    @classmethod
    @lru_cache
    def from_url(cls, url: str) -> PDF:
        parsed_url = urlparse(url)
        media_type, _ = mimetypes.guess_type(parsed_url.path)

        if not media_type:
            try:
                response = requests.head(url, allow_redirects=True)
                media_type = response.headers.get("Content-Type")
            except requests.RequestException as e:
                raise ValueError("Failed to fetch PDF from URL") from e

        if media_type not in VALID_PDF_MIME_TYPES:
            raise ValueError(f"Unsupported PDF format: {media_type}")
        return cls(source=url, media_type=media_type, data=None)

    def to_mistral(self) -> dict[str, Any]:
        if (
            isinstance(self.source, str)
            and self.source.startswith(("http://", "https://"))
            and not self.data
        ):
            return {
                "type": "document_url",
                "document_url": self.source,
            }
        raise ValueError("Mistral only supports document URLs for now")

    def to_openai(self, mode: Mode) -> dict[str, Any]:
        """Convert to OpenAI's document format."""
        input_file_type = (
            "input_file"
            if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}
            else "file"
        )

        if (
            isinstance(self.source, str)
            and self.source.startswith(("http://", "https://"))
            and not self.data
        ):
            # Fetch the file from URL and convert to base64
            data = requests.get(self.source)
            data = base64.b64encode(data.content).decode("utf-8")
            if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}:
                return {
                    "type": input_file_type,
                    "filename": self.source,
                    "file_data": f"data:{self.media_type};base64,{data}",
                }
            else:
                return {
                    "type": input_file_type,
                    "file": {
                        "filename": self.source,
                        "file_data": f"data:{self.media_type};base64,{data}",
                    },
                }
        elif self.data or self.is_base64(str(self.source)):
            data = self.data or str(self.source).split(",", 1)[1]
            if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}:
                return {
                    "type": input_file_type,
                    "filename": self.source
                    if isinstance(self.source, str)
                    else str(self.source),
                    "file_data": f"data:{self.media_type};base64,{data}",
                }
            else:
                return {
                    "type": input_file_type,
                    "file": {
                        "filename": self.source
                        if isinstance(self.source, str)
                        else str(self.source),
                        "file_data": f"data:{self.media_type};base64,{data}",
                    },
                }
        else:
            raise ValueError("PDF data is missing for base64 encoding.")

    def to_anthropic(self) -> dict[str, Any]:
        """Convert to Anthropic's document format."""
        if (
            isinstance(self.source, str)
            and self.source.startswith(("http://", "https://"))
            and not self.data
        ):
            return {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": self.source,
                },
            }
        else:
            if not self.data:
                self.data = requests.get(str(self.source)).content  # type: ignore
                self.data = base64.b64encode(self.data).decode("utf-8")  # type: ignore

            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": self.media_type,
                    "data": self.data,
                },
            }

    def to_genai(self):
        from google.genai import types

        if (
            isinstance(self.source, str)
            and self.source.startswith(("http://", "https://"))
            and not self.data
        ):
            # Fetch the file from URL and convert to base64
            data = requests.get(self.source).content
            data = base64.b64encode(data).decode("utf-8")
            return types.Part.from_bytes(
                data=base64.b64decode(data),
                mime_type=self.media_type,
            )

        if self.data:
            return types.Part.from_bytes(
                data=base64.b64decode(self.data),
                mime_type=self.media_type,
            )

        raise ValueError("Unsupported PDF format")


class PDFWithCacheControl(PDF):
    """PDF with Anthropic prompt caching support."""

    def to_anthropic(self) -> dict[str, Any]:
        """Override Anthropic return with cache_control."""
        result = super().to_anthropic()
        result["cache_control"] = {"type": "ephemeral"}
        return result


class PDFWithGenaiFile(PDF):
    @classmethod
    def from_new_genai_file(
        cls, file_path: str, retry_delay: int = 10, max_retries: int = 20
    ) -> PDFWithGenaiFile:
        """Create a new PDFWithGenaiFile from a file path."""
        from google.genai.types import FileState
        import time
        from google.genai import Client

        client = Client()
        file = client.files.upload(file=file_path)
        while file.state != FileState.ACTIVE:
            time.sleep(retry_delay)
            file = client.files.get(name=file.name)  # type: ignore
            if max_retries > 0:
                max_retries -= 1
            else:
                raise Exception(
                    "Max retries reached. File upload has been started but is still pending"
                )

        return cls(source=file.uri, media_type=file.mime_type, data=None)  # type: ignore

    @classmethod
    def from_existing_genai_file(cls, file_name: str) -> PDFWithGenaiFile:
        """Create a new PDFWithGenaiFile from a file URL."""
        from google.genai import types
        from google.genai.types import FileState
        from google.genai import Client

        client = Client()
        file = client.files.get(name=file_name)
        if file.source == types.FileSource.UPLOADED and file.state == FileState.ACTIVE:
            return cls(
                source=file.uri,  # type: ignore
                media_type=file.mime_type,  # type: ignore
                data=None,
            )
        else:
            raise ValueError("We only support uploaded PDFs for now")

    def to_genai(self):
        from google.genai import types

        if (
            self.source
            and isinstance(self.source, str)
            and "https://generativelanguage.googleapis.com/v1beta/files/" in self.source
        ):
            return types.Part.from_uri(
                file_uri=self.source,
                mime_type=self.media_type,
            )

        return super().to_genai()


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
    if isinstance(contents, (Image, Audio, PDF)) or isinstance(contents, dict):
        contents = [contents]

    converted_contents: list[dict[str, Union[str, Image]]] = []  # noqa: UP007
    text_file_type = (
        "input_text"
        if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}
        else "text"
    )
    for content in contents:
        if isinstance(content, str):
            converted_contents.append({"type": text_file_type, "text": content})
        elif isinstance(content, dict):
            converted_contents.append(content)
        elif isinstance(content, (Image, Audio, PDF)):
            if mode in {
                Mode.ANTHROPIC_JSON,
                Mode.ANTHROPIC_TOOLS,
                Mode.ANTHROPIC_REASONING_TOOLS,
            }:
                converted_contents.append(content.to_anthropic())
            elif mode in {Mode.GEMINI_JSON, Mode.GEMINI_TOOLS}:
                raise NotImplementedError("Gemini is not supported yet")
            elif mode in {
                Mode.MISTRAL_STRUCTURED_OUTPUTS,
                Mode.MISTRAL_TOOLS,
            } and isinstance(content, (PDF)):
                converted_contents.append(content.to_mistral())  # type: ignore
            else:
                converted_contents.append(content.to_openai(mode))
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


def extract_genai_multimodal_content(
    contents: list[Any],
    autodetect_images: bool = True,
):
    """
    Convert Typed Contents to the appropriate format for Google GenAI.
    """
    from google.genai import types

    result: list[Union[types.Content, types.File]] = []  # noqa: UP007
    for content in contents:
        # Check for Files
        if isinstance(content, types.File):
            result.append(content)
            continue

        # We only want to do the conversion for the Image type
        if not isinstance(content, types.Content):
            raise ValueError(
                f"Unsupported content type: {type(content)}. This should only be used for the Google types"
            )
        # Cast to list of Parts
        content = cast(types.Content, content)
        converted_contents: list[types.Part] = []

        if not content.parts:
            raise ValueError("Content parts are empty")

        # Now we need to support a few cases
        for content_part in content.parts:
            if content_part.text and autodetect_images:
                # Detect if the text is an image
                converted_item = Image.autodetect_safely(content_part.text)

                # We only do autodetection for images for now
                if isinstance(converted_item, Image):
                    converted_contents.append(converted_item.to_genai())
                    continue

                # If it's not an image or audio, we just return the text
                converted_contents.append(content_part)
            else:
                converted_contents.append(content_part)

        result.append(types.Content(parts=converted_contents, role=content.role))

    return result
