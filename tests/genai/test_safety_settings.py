from instructor.providers.gemini.utils import update_genai_kwargs


def test_update_genai_kwargs_safety_settings_with_image_content_uses_image_categories():
    """Image inputs should use IMAGE_* harm categories when available."""
    from google.genai import types
    from google.genai.types import HarmCategory

    excluded_categories = {HarmCategory.HARM_CATEGORY_UNSPECIFIED}
    if hasattr(HarmCategory, "HARM_CATEGORY_JAILBREAK"):
        excluded_categories.add(HarmCategory.HARM_CATEGORY_JAILBREAK)

    image_categories = [
        c
        for c in HarmCategory
        if c not in excluded_categories and c.name.startswith("HARM_CATEGORY_IMAGE_")
    ]

    # Older SDKs may not expose separate image categories.
    if not image_categories:
        return

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
    assert len(result["safety_settings"]) == len(image_categories)
    assert {s["category"] for s in result["safety_settings"]} == set(image_categories)


def test_update_genai_kwargs_maps_text_thresholds_to_image_categories():
    """Text thresholds should carry over to equivalent IMAGE_* categories."""
    from google.genai import types
    from google.genai.types import HarmBlockThreshold, HarmCategory

    excluded_categories = {HarmCategory.HARM_CATEGORY_UNSPECIFIED}
    if hasattr(HarmCategory, "HARM_CATEGORY_JAILBREAK"):
        excluded_categories.add(HarmCategory.HARM_CATEGORY_JAILBREAK)

    image_categories = [
        c
        for c in HarmCategory
        if c not in excluded_categories and c.name.startswith("HARM_CATEGORY_IMAGE_")
    ]

    if not image_categories or not hasattr(HarmCategory, "HARM_CATEGORY_IMAGE_HATE"):
        return

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
        if setting["category"] == HarmCategory.HARM_CATEGORY_IMAGE_HATE:
            assert setting["threshold"] == HarmBlockThreshold.BLOCK_LOW_AND_ABOVE


def test_handle_genai_tools_autodetect_images_uses_image_categories():
    """Autodetected image content should switch safety_settings to IMAGE_* categories."""
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
    assert any(
        s.category.name.startswith("HARM_CATEGORY_IMAGE_")
        for s in out["config"].safety_settings
    )
