import pytest
from pathlib import Path
from instructor.multimodal import Image, convert_contents, convert_messages
from instructor.mode import Mode


def test_image_from_url():
    url = "https://example.com/image.jpg"
    image = Image.from_url(url)
    assert image.source == url
    assert image.media_type == "image/jpeg"
    assert image.data is None


def test_image_from_path(tmp_path: Path):
    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(b"fake image data")

    image = Image.from_path(image_path)
    assert image.source == str(image_path)
    assert image.media_type == "image/jpeg"
    assert image.data is not None


@pytest.mark.skip(reason="Needs to download image")
def test_image_to_anthropic():
    image = Image(
        source="http://example.com/image.jpg", media_type="image/jpeg", data=None
    )
    anthropic_format = image.to_anthropic()
    assert anthropic_format["type"] == "image"
    assert anthropic_format["source"]["type"] == "base64"
    assert anthropic_format["source"]["media_type"] == "image/jpeg"


def test_image_to_openai():
    image = Image(
        source="http://example.com/image.jpg", media_type="image/jpeg", data=None
    )
    openai_format = image.to_openai()
    assert openai_format["type"] == "image_url"
    assert openai_format["image_url"]["url"] == "http://example.com/image.jpg"


def test_convert_contents():
    contents = ["Hello", Image.from_url("http://example.com/image.jpg")]
    converted = list(convert_contents(contents, Mode.TOOLS))
    assert len(converted) == 2
    assert converted[0] == {"type": "text", "text": "Hello"}
    assert converted[1]["type"] == "image_url"
    assert converted[1]["image_url"]["url"] == "http://example.com/image.jpg"


def test_convert_messages():
    messages = [
        {
            "role": "user",
            "content": ["Hello", Image.from_url("http://example.com/image.jpg")],
        },
        {"role": "assistant", "content": "Hi there!"},
    ]
    converted = list(convert_messages(messages, Mode.TOOLS))
    assert len(converted) == 2
    assert converted[0]["role"] == "user"
    assert len(converted[0]["content"]) == 2
    assert converted[0]["content"][0] == {"type": "text", "text": "Hello"}
    assert converted[0]["content"][1]["type"] == "image_url"
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"] == "Hi there!"


def test_convert_messages_anthropic():
    messages = [
        {
            "role": "user",
            "content": [
                "Hello",
                Image(source="base64data", media_type="image/jpeg", data="fakedata"),
            ],
        }
    ]
    converted = list(convert_messages(messages, Mode.ANTHROPIC_JSON))
    assert len(converted) == 1
    assert converted == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "fakedata",
                    },
                },
            ],
        }
    ]


def test_convert_messages_gemini():
    messages = [
        {
            "role": "user",
            "content": ["Hello", Image.from_url("http://example.com/image.jpg")],
        }
    ]
    with pytest.raises(NotImplementedError):
        list(convert_messages(messages, Mode.GEMINI_JSON))


# Additional tests


def test_image_from_path_unsupported_format(tmp_path: Path):
    image_path = tmp_path / "test_image.gif"
    image_path.write_bytes(b"fake gif data")

    with pytest.raises(ValueError, match="Unsupported image format: gif"):
        Image.from_path(image_path)


def test_image_from_path_empty_file(tmp_path: Path):
    image_path = tmp_path / "empty_image.jpg"
    image_path.touch()

    with pytest.raises(ValueError, match="Image file is empty"):
        Image.from_path(image_path)


def test_image_to_openai_base64():
    image = Image(
        source="local_file.jpg", media_type="image/jpeg", data="base64encodeddata"
    )
    openai_format = image.to_openai()
    assert openai_format["type"] == "image_url"
    assert openai_format["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_convert_contents_single_string():
    content = "Hello, world!"
    converted = convert_contents(content, Mode.TOOLS)
    assert converted == "Hello, world!"


def test_convert_contents_single_image():
    image = Image.from_url("http://example.com/image.jpg")
    converted = list(convert_contents(image, Mode.TOOLS))
    assert len(converted) == 1
    assert converted == [
        {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
    ]


def test_convert_messages_mixed_content():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": Image.from_url("http://example.com/image.jpg")},
    ]
    converted = list(convert_messages(messages, Mode.TOOLS))
    assert len(converted) == 3
    assert converted[0]["content"] == "Hello"
    assert converted[1]["content"] == "Hi there!"
    assert converted[2]["content"][0]["type"] == "image_url"


def test_convert_contents_invalid_type():
    with pytest.raises(ValueError, match="Unsupported content type"):
        list(convert_contents([1, 2, 3], Mode.TOOLS))


def test_convert_contents_anthropic_mode():
    contents = [
        "Hello",
        Image(source="base64data", media_type="image/png", data="fakedata"),
    ]
    converted = list(convert_contents(contents, Mode.ANTHROPIC_JSON))
    assert converted[1]["type"] == "image"
    assert converted[1]["source"]["type"] == "base64"
    assert converted[1]["source"]["media_type"] == "image/png"
