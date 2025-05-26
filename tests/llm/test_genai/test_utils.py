from instructor.utils import update_genai_kwargs


def test_update_genai_kwargs_basic():
    """Test basic parameter mapping from OpenAI to Gemini format."""
    kwargs = {
        "generation_config": {
            "max_tokens": 100,
            "temperature": 0.7,
            "n": 2,
            "top_p": 0.9,
            "stop": ["END"],
            "seed": 42,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
        }
    }
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Check that OpenAI parameters were mapped to Gemini equivalents
    assert result["max_output_tokens"] == 100
    assert result["temperature"] == 0.7
    assert result["candidate_count"] == 2
    assert result["top_p"] == 0.9
    assert result["stop_sequences"] == ["END"]
    assert result["seed"] == 42
    assert result["presence_penalty"] == 0.1
    assert result["frequency_penalty"] == 0.2


def test_update_genai_kwargs_safety_settings():
    """Test that safety settings are properly configured."""
    from google.genai.types import HarmCategory, HarmBlockThreshold

    kwargs = {}
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Check that safety_settings is configured as a list
    assert "safety_settings" in result
    assert isinstance(result["safety_settings"], list)

    # Should have one entry for each HarmCategory
    assert len(result["safety_settings"]) == (len(HarmCategory) - 1)

    # Each entry should be a dict with category and threshold
    for setting in result["safety_settings"]:
        assert isinstance(setting, dict)
        assert "category" in setting
        assert "threshold" in setting
        assert setting["threshold"] == HarmBlockThreshold.BLOCK_NONE  # Default


def test_update_genai_kwargs_with_custom_safety_settings():
    """Test that custom safety settings are properly handled."""
    from google.genai.types import HarmCategory, HarmBlockThreshold

    # Test with one category that exists in safety_settings
    custom_safety = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    }
    kwargs = {"safety_settings": custom_safety}
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Check that safety_settings is configured as a list
    assert "safety_settings" in result
    assert isinstance(result["safety_settings"], list)

    # Should have one entry for each HarmCategory
    assert len(result["safety_settings"]) == (len(HarmCategory) - 1)

    for setting in result["safety_settings"]:
        if setting["category"] == HarmCategory.HARM_CATEGORY_HATE_SPEECH:
            assert setting["threshold"] == HarmBlockThreshold.BLOCK_LOW_AND_ABOVE

    # Other categories should use the default
    for setting in result["safety_settings"]:
        if setting["category"] != HarmCategory.HARM_CATEGORY_HATE_SPEECH:
            assert setting["threshold"] == HarmBlockThreshold.BLOCK_NONE


def test_update_genai_kwargs_none_values():
    """Test that None values are not set in the result."""
    kwargs = {
        "generation_config": {
            "max_tokens": None,
            "temperature": 0.7,
            "n": None,
        }
    }
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Check that None values are not included
    assert "max_output_tokens" not in result
    assert "candidate_count" not in result
    assert result["temperature"] == 0.7


def test_update_genai_kwargs_empty():
    """Test with empty kwargs."""
    kwargs = {}
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # Should still have safety_settings configured
    assert "safety_settings" in result


def test_update_genai_kwargs_preserves_original():
    """Test that the function doesn't modify the original kwargs."""
    original_kwargs = {
        "generation_config": {
            "max_tokens": 100,
            "temperature": 0.7,
        },
        "safety_settings": {},
    }
    kwargs = original_kwargs.copy()
    base_config = {}

    result = update_genai_kwargs(kwargs, base_config)

    # The function should not modify the original kwargs (works on a copy)
    assert kwargs == original_kwargs
    # But result should have the mapped parameters
    assert "max_output_tokens" in result
    assert "temperature" in result
