import instructor
import pytest
from pydantic import BaseModel
import os
from .util import models, modes
from instructor.multimodal import Image


class ImageDescription(BaseModel):
    items: list[str]


curr_file = os.path.dirname(__file__)
file_path = os.path.join(curr_file, "../../assets/image.jpg")

print(file_path)


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_local_file_image(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "What is shown in this image?",
                    instructor.Image.from_path(file_path),
                ],
            }
        ],
        response_model=ImageDescription,
    )
    assert isinstance(response, ImageDescription)
    assert len(response.items) > 0


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_remote_url_image(client, model, mode):
    client = instructor.from_genai(client, mode=mode)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "Describe this image in detail",
                    "gs://generativeai-downloads/images/scones.jpg",
                ],
            }
        ],
        response_model=ImageDescription,
    )
    assert isinstance(response, ImageDescription)
    assert len(response.items) > 0


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_gs_url_image(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    image_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": ["What do you see in this image?", image_url]}
        ],
        response_model=ImageDescription,
    )
    assert isinstance(response, ImageDescription)
    assert len(response.items) > 0


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_instructor_image(client, model, mode):
    client = instructor.from_genai(client, mode=mode)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": ["Analyze this image", file_path],
            }
        ],
        response_model=ImageDescription,
    )
    assert isinstance(response, ImageDescription)
    assert len(response.items) > 0
