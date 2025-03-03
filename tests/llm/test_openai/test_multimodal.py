import pytest
from instructor.multimodal import Image, Audio
import instructor
from pydantic import Field, BaseModel
from itertools import product
from .util import models, modes
import requests
from pathlib import Path


audio_url = "https://www.cs.uic.edu/~i101/SoundFiles/gettysburg.wav"
image_url = "https://retail.degroot-inc.com/wp-content/uploads/2024/01/AS_Blueberry_Patriot_1-605x605.jpg"


def gettysburg_audio():
    audio_file = Path("gettysburg.wav")
    if not audio_file.exists():
        response = requests.get(audio_url)
        response.raise_for_status()
        with open(audio_file, "wb") as f:
            f.write(response.content)
    return audio_file


@pytest.mark.parametrize(
    "audio_file",
    [Audio.from_url(audio_url), Audio.from_path(gettysburg_audio())],
)
def test_multimodal_audio_description(audio_file, client):
    client = instructor.from_openai(client)

    class AudioDescription(BaseModel):
        source: str

    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        response_model=AudioDescription,
        modalities=["text"],
        messages=[
            {
                "role": "user",
                "content": [
                    "Where's this excerpt from?",
                    audio_file,
                ],
            },
        ],
        audio={"voice": "alloy", "format": "wav"},
    )


class ImageDescription(BaseModel):
    objects: list[str] = Field(..., description="The objects in the image")
    scene: str = Field(..., description="The scene of the image")
    colors: list[str] = Field(..., description="The colors in the image")


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description(model, mode, client):
    client = instructor.from_openai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,  # Ensure this is a vision-capable model
        response_model=ImageDescription,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images",
            },
            {
                "role": "user",
                "content": [
                    "What is this?",
                    Image.from_url(image_url),
                ],
            },
        ],
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0

    # Additional assertions can be added based on expected content of the sample image


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect(model, mode, client):
    client = instructor.from_openai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,  # Ensure this is a vision-capable model
        response_model=ImageDescription,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images",
            },
            {
                "role": "user",
                "content": [
                    "What is this?",
                    image_url,
                ],
            },
        ],
        autodetect_images=True,
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0

    # Additional assertions can be added based on expected content of the sample image


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect_no_response_model(model, mode, client):
    client = instructor.from_openai(client, mode=mode)
    response = client.chat.completions.create(
        response_model=None,
        model=model,  # Ensure this is a vision-capable model
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images. "
                "If looking at an image, reply with 'This is an image' and nothing else.",
            },
            {
                "role": "user",
                "content": image_url,
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    assert response.choices[0].message.content.startswith("This is an image")
