import pytest
from instructor.multimodal import Image, PDF, PDFWithCacheControl
import instructor
from pydantic import Field, BaseModel
from itertools import product
from .util import models, modes
import os
import base64


class ImageDescription(BaseModel):
    objects: list[str] = Field(..., description="The objects in the image")
    scene: str = Field(..., description="The scene of the image")
    colors: list[str] = Field(..., description="The colors in the image")


image_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"

pdf_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"


curr_file = os.path.dirname(__file__)
pdf_path = os.path.join(curr_file, "../../assets/invoice.pdf")
pdf_base64 = base64.b64encode(open(pdf_path, "rb").read()).decode("utf-8")
pdf_base64_string = f"data:application/pdf;base64,{pdf_base64}"


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description(model, mode, client):
    client = instructor.from_anthropic(client, mode=mode)
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
        temperature=1,
        max_tokens=1000,
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect(model, mode, client):
    client = instructor.from_anthropic(client, mode=mode)
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
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0

    # Additional assertions can be added based on expected content of the sample image


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect_image_params(model, mode, client):
    client = instructor.from_anthropic(client, mode=mode)
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
                    {
                        "type": "image",
                        "source": image_url,
                    },
                ],
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0

    # Additional assertions can be added based on expected content of the sample image


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect_image_params_cache(
    model, mode, client
):
    client = instructor.from_anthropic(client, mode=mode)
    messages = client.chat.completions.create(
        model=model,  # Ensure this is a vision-capable model
        response_model=None,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images and stuff",
            },
            {
                "role": "user",
                "content": [
                    "Describe these images",
                    # Large images to activate caching
                    {
                        "type": "image",
                        "source": "https://assets.entrepreneur.com/content/3x2/2000/20200429211042-GettyImages-1164615296.jpeg",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "image",
                        "source": "https://www.bigbear.com/imager/s3_us-west-1_amazonaws_com/big-bear/images/Scenic-Snow/89xVzXp1_00588cdef1e3d54756582b576359604b.jpeg",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    # Assert a cache write or cache hit
    assert (
        messages.usage.cache_creation_input_tokens > 0
        or messages.usage.cache_read_input_tokens > 0
    )


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect_no_response_model(model, mode, client):
    client = instructor.from_anthropic(client, mode=mode)
    system_message = (
        "You are a helpful assistant that can describe images. "
        "If looking at an image, reply with 'This is an image' and nothing else."
    )
    # Test with OpenAI style messages
    response = client.chat.completions.create(
        response_model=None,
        model=model,  # Ensure this is a vision-capable model
        messages=[
            {
                "role": "system",
                "content": system_message,
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

    assert response.content[0].text.startswith("This is an image")

    # Test with Anthropic style messages
    response = client.chat.completions.create(
        response_model=None,
        model=model,  # Ensure this is a vision-capable model
        system=system_message,
        messages=[
            {
                "role": "user",
                "content": image_url,
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    assert response.content[0].text.startswith("This is an image")


class LineItem(BaseModel):
    name: str
    price: int
    quantity: int


class Receipt(BaseModel):
    total: int
    items: list[str]


@pytest.mark.parametrize("pdf_source", [pdf_path, pdf_url, pdf_base64_string])
@pytest.mark.parametrize("mode", modes)
def test_multimodal_pdf_file(mode, client, pdf_source):
    client = instructor.from_anthropic(client, mode=mode)
    response = client.chat.completions.create(
        model="claude-3-5-sonnet-20240620",  # Ensure this is a vision-capable model
        messages=[
            {
                "role": "system",
                "content": "Extract the total and items from the invoice",
            },
            {
                "role": "user",
                "content": PDF.autodetect(pdf_source),
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=False,
        response_model=Receipt,
    )

    assert response.total == 220
    assert len(response.items) == 2


@pytest.mark.parametrize("pdf_source", [pdf_path, pdf_url, pdf_base64_string])
@pytest.mark.parametrize("mode", modes)
def test_multimodal_pdf_file_with_cache_control(mode, client, pdf_source):
    client = instructor.from_anthropic(client, mode=mode)

    response, completion = client.chat.completions.create_with_completion(
        model="claude-3-5-sonnet-20240620",  # Ensure this is a vision-capable model
        messages=[
            {
                "role": "system",
                "content": "Extract the total and items from the invoice",
            },
            {
                "role": "user",
                "content": PDFWithCacheControl.autodetect(pdf_source),
            },
        ],
        max_tokens=1000,
        autodetect_images=False,
        response_model=Receipt,
    )

    assert response.total == 220
    assert (
        completion.usage.cache_creation_input_tokens > 0
        or completion.usage.cache_read_input_tokens > 0
    )
    assert len(response.items) == 2
