import pytest


def test_from_provider_xai_requires_optional_extra():
    import instructor
    from instructor.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError) as excinfo:
        instructor.from_provider("xai/grok-3-mini", api_key="test-key")

    msg = str(excinfo.value)
    assert "instructor[xai]" in msg
    assert "uv pip install" in msg


def test_direct_from_xai_has_clear_error_when_sdk_missing():
    from instructor.core.exceptions import ConfigurationError
    from instructor.providers.xai.client import from_xai

    with pytest.raises(ConfigurationError) as excinfo:
        from_xai(object())  # type: ignore[arg-type]

    msg = str(excinfo.value)
    assert "instructor[xai]" in msg
    assert "xai-sdk" in msg

