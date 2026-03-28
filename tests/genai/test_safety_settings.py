from instructor.providers.gemini.utils import update_genai_kwargs


def test_update_genai_kwargs_excludes_image_categories_for_gemini_api():
    """IMAGE_* harm categories are Vertex AI-only and must not be sent to the Gemini API.

    See: https://github.com/instructor-ai/instructor/issues/2146
    """
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
    base_config: dict = {}

    result = update_genai_kwargs(kwargs, base_config)

    assert "safety_settings" in result
    assert isinstance(result["safety_settings"], list)
    # No IMAGE_* categories should be present -- they cause 400 on the Gemini API
    for setting in result["safety_settings"]:
        assert not setting["category"].name.startswith("HARM_CATEGORY_IMAGE_"), (
            f"IMAGE_ category {setting['category'].name} should not be sent to Gemini API"
        )


def test_update_genai_kwargs_includes_text_categories_for_image_content():
    """Even when the request contains images, only text categories should be used."""
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

    returned_categories = {s["category"] for s in result["safety_settings"]}
    assert returned_categories == set(text_categories)


def test_update_genai_kwargs_respects_custom_thresholds():
    """User-provided thresholds should be applied to text categories."""
    from google.genai.types import HarmBlockThreshold, HarmCategory

    custom_safety = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    }

    kwargs: dict = {"safety_settings": custom_safety}
    base_config: dict = {}

    result = update_genai_kwargs(kwargs, base_config)

    for setting in result["safety_settings"]:
        if setting["category"] == HarmCategory.HARM_CATEGORY_HATE_SPEECH:
            assert setting["threshold"] == HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            break
    else:
        raise AssertionError("HARM_CATEGORY_HATE_SPEECH not found in safety settings")


def test_handle_genai_tools_excludes_image_categories():
    """handle_genai_tools must not include IMAGE_* categories even with image content.

    Regression test for https://github.com/instructor-ai/instructor/issues/2146
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
    assert not any(
        s.category.name.startswith("HARM_CATEGORY_IMAGE_")
        for s in out["config"].safety_settings
    ), "IMAGE_ categories should not be sent to the Gemini API"
