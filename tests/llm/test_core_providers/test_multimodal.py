"""
Multimodal tests that run across all core providers.

Tests unified multimodal API for images across providers that support it.
Provider-specific features (like audio, PDF cache control) remain in provider-specific tests.
"""

from pydantic import BaseModel, Field
import pytest
import instructor
from instructor.processing.multimodal import Image


class ImageDescription(BaseModel):
    """Description of an image including objects, scene, and colors."""

    objects: list[str] = Field(..., description="The objects in the image")
    scene: str = Field(..., description="The scene/setting of the image")
    colors: list[str] = Field(..., description="The dominant colors in the image")


# Test image URL - a simple image that should work across all providers
IMAGE_URL = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"


def test_multimodal_image_from_url(provider_config):
    """Test image description using Image.from_url() - should work across all providers."""
    model, mode = provider_config

    # Skip providers that don't support multimodal
    if "ministral" in model.lower() or "palmyra" in model.lower():
        pytest.skip(f"Model {model} doesn't support multimodal")

    client = instructor.from_provider(model, mode=mode)

    response = client.chat.completions.create(
        response_model=ImageDescription,
        messages=[
            {
                "role": "user",
                "content": [
                    "What do you see in this image? Describe the objects, scene, and colors.",
                    Image.from_url(IMAGE_URL),
                ],
            },
        ],
        max_tokens=1000,
        temperature=0.5,
    )

    # Validate response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0, "Should identify at least one object"
    assert response.scene, "Should describe the scene"
    assert len(response.colors) > 0, "Should identify at least one color"


def test_multimodal_image_autodetect(provider_config):
    """Test image autodetection from URL string - unified across providers."""
    model, mode = provider_config

    # Skip providers that don't support multimodal or autodetect
    if "ministral" in model.lower() or "palmyra" in model.lower():
        pytest.skip(f"Model {model} doesn't support multimodal")

    # Skip Anthropic as it requires explicit Image objects
    if "anthropic" in model.lower():
        pytest.skip("Anthropic requires explicit Image.from_url(), doesn't support autodetect")

    client = instructor.from_provider(model, mode=mode)

    response = client.chat.completions.create(
        response_model=ImageDescription,
        messages=[
            {
                "role": "user",
                "content": [
                    "What do you see in this image?",
                    IMAGE_URL,  # Just the URL string, no Image wrapper
                ],
            },
        ],
        max_tokens=1000,
        temperature=0.5,
        autodetect_images=True,
    )

    # Validate response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene
    assert len(response.colors) > 0


@pytest.mark.asyncio
async def test_async_multimodal_image(provider_config):
    """Test async image description - should work across all providers."""
    model, mode = provider_config

    # Skip providers that don't support multimodal
    if "ministral" in model.lower() or "palmyra" in model.lower():
        pytest.skip(f"Model {model} doesn't support multimodal")

    client = instructor.from_provider(model, mode=mode, async_client=True)

    response = await client.chat.completions.create(
        response_model=ImageDescription,
        messages=[
            {
                "role": "user",
                "content": [
                    "Describe what you see in this image.",
                    Image.from_url(IMAGE_URL),
                ],
            },
        ],
        max_tokens=1000,
        temperature=0.5,
    )

    # Validate response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene
    assert len(response.colors) > 0


class SimpleImageLabel(BaseModel):
    """Simple yes/no image classification."""

    contains_person: bool = Field(..., description="Whether image contains a person")
    is_outdoor: bool = Field(..., description="Whether image is outdoors")


def test_multimodal_image_boolean_fields(provider_config):
    """Test boolean classification with images - unified API."""
    model, mode = provider_config

    # Skip providers that don't support multimodal
    if "ministral" in model.lower() or "palmyra" in model.lower():
        pytest.skip(f"Model {model} doesn't support multimodal")

    client = instructor.from_provider(model, mode=mode)

    response = client.chat.completions.create(
        response_model=SimpleImageLabel,
        messages=[
            {
                "role": "user",
                "content": [
                    "Answer these questions about the image:",
                    Image.from_url(IMAGE_URL),
                ],
            },
        ],
        max_tokens=100,
        temperature=0,
    )

    # Validate response structure (can't validate specific answers without knowing image)
    assert isinstance(response, SimpleImageLabel)
    assert isinstance(response.contains_person, bool)
    assert isinstance(response.is_outdoor, bool)
