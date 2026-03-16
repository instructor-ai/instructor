from instructor.providers.gemini.utils import update_genai_kwargs


def test_update_genai_kwargs_safety_settings_with_image_content_uses_text_categories():
    """Image inputs should use text harm categories for Gemini API (not IMAGE_*).
    
    IMAGE_* harm categories are only supported by Vertex AI API, not the standard
    Gemini API. See: https://github.com/instructor-ai/instructor/issues/2146
    """
    from google.genai import types
    from google.genai.types import HarmCategory

    excluded_categories = {HarmCategory.HARM_CATEGORY_UNSPECIFIED}
    if hasattr(HarmCategory, "HARM_CATEGORY_JAILBREAK"):
        excluded_categories.add(HarmCategory.HARM_CATEGORY_JAILBREAK)

    text_categories = [
        c
        for c in HarmCategory
        if c not in excluded_categories
        and not c.name.startswith("HARM_CATEGORY_IMAGE_")
    ]

    kwargs = {
        "contents": [
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=b"123", mime_type="image/png")],
            )
        ]
    }
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    assert "safety_settings" in result
    assert isinstance(result["safety_settings"], list)
    # Should use text categories, not IMAGE_* categories
    assert len(result["safety_settings"]) == len(text_categories)
    assert {s["category"] for s in result["safety_settings"]} == set(text_categories)


def test_update_genai_kwargs_safety_settings_does_not_use_image_categories():
    """Gemini API should never use IMAGE_* harm categories."""
    from google.genai import types
    from google.genai.types import HarmCategory

    kwargs = {
        "contents": [
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=b"123", mime_type="image/png")],
            )
        ]
    }
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Ensure no IMAGE_* categories are used
    for setting in result["safety_settings"]:
        assert not setting["category"].name.startswith("HARM_CATEGORY_IMAGE_")


def test_update_genai_kwargs_maps_custom_safety_settings():
    """Custom safety settings should be applied to text categories."""
    from google.genai import types
    from google.genai.types import HarmBlockThreshold, HarmCategory

    custom_safety = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    }

    kwargs = {
        "contents": [
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=b"123", mime_type="image/png")],
            )
        ],
        "safety_settings": custom_safety,
    }
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    for setting in result["safety_settings"]:
        if setting["category"] == HarmCategory.HARM_CATEGORY_HATE_SPEECH:
            assert setting["threshold"] == HarmBlockThreshold.BLOCK_LOW_AND_ABOVE


def test_handle_genai_tools_autodetect_images_uses_text_categories():
    """Autodetected image content should use text harm categories for Gemini API.
    
    IMAGE_* harm categories are only supported by Vertex AI API.
    """
    from pydantic import BaseModel

    from instructor.providers.gemini.utils import handle_genai_tools

    class SimpleModel(BaseModel):
        text: str

    data_uri = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6q0S8AAAAASUVORK5CYII="
    )

    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": ["What is in this image?", data_uri],
            }
        ]
    }

    _, out = handle_genai_tools(SimpleModel, kwargs, autodetect_images=True)

    assert "config" in out
    assert out["config"].safety_settings is not None
    # Should use text categories, not IMAGE_* categories
    assert all(
        not s.category.name.startswith("HARM_CATEGORY_IMAGE_")
        for s in out["config"].safety_settings
    )
