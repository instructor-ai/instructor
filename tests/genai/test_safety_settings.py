from instructor.providers.gemini.utils import update_genai_kwargs


def test_update_genai_kwargs_always_uses_text_categories_even_with_image_content():
    """Image inputs must still use text harm categories on Google GenAI (not Vertex AI).

    HARM_CATEGORY_IMAGE_* categories are only valid on Vertex AI.
    update_genai_kwargs is exclusively called from GenAI handlers,
    so it must never emit IMAGE_* categories.  See issue #2146.
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
    base_config: dict = {}

    result = update_genai_kwargs(kwargs, base_config)

    assert "safety_settings" in result
    assert isinstance(result["safety_settings"], list)
    assert len(result["safety_settings"]) == len(text_categories)
    emitted = {s["category"] for s in result["safety_settings"]}
    assert emitted == set(text_categories)
    # Ensure no IMAGE_* category leaked through
    for s in result["safety_settings"]:
        assert not s["category"].name.startswith("HARM_CATEGORY_IMAGE_"), (
            f"IMAGE_* category {s['category']} should not appear in GenAI safety settings"
        )


def test_update_genai_kwargs_text_thresholds_applied_with_image_content():
    """User-supplied text thresholds should be respected even when images are present."""
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
    base_config: dict = {}

    result = update_genai_kwargs(kwargs, base_config)

    for setting in result["safety_settings"]:
        if setting["category"] == HarmCategory.HARM_CATEGORY_HATE_SPEECH:
            assert setting["threshold"] == HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            break
    else:
        raise AssertionError("HARM_CATEGORY_HATE_SPEECH not found in safety_settings")


def test_handle_genai_tools_autodetect_images_uses_text_categories():
    """Autodetected image content must still use text categories on GenAI."""
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
    # All emitted categories must be text categories, not IMAGE_*
    for s in out["config"].safety_settings:
        assert not s.category.name.startswith("HARM_CATEGORY_IMAGE_"), (
            f"IMAGE_* category {s.category} should not appear in GenAI safety settings"
        )
