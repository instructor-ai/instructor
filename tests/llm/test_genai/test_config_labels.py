from pydantic import BaseModel


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


def test_config_labels_preserved_unit():
    """Unit test to verify config labels are preserved in handle_genai functions."""
    from instructor.providers.gemini.utils import handle_genai_structured_outputs, handle_genai_tools
    
    test_labels = {
        "environment": "development",
        "tenant": "test_tenant"
    }
    
    response_model = SimpleResponse
    kwargs_with_labels = {
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "config": {"labels": test_labels}
    }
    
    result_model, result_kwargs = handle_genai_structured_outputs(
        response_model, kwargs_with_labels.copy()
    )
    
    assert "config" in result_kwargs
    config = result_kwargs["config"]
    assert hasattr(config, 'labels'), "Config should have labels attribute"
    assert config.labels == test_labels, f"Labels should be preserved in structured outputs. Expected: {test_labels}, Got: {config.labels}"
    
    result_model, result_kwargs = handle_genai_tools(
        response_model, kwargs_with_labels.copy()
    )
    
    assert "config" in result_kwargs
    config = result_kwargs["config"]
    assert hasattr(config, 'labels'), "Config should have labels attribute"
    assert config.labels == test_labels, f"Labels should be preserved in tools mode. Expected: {test_labels}, Got: {config.labels}"


def test_config_labels_with_other_config_params():
    """Test that labels work alongside other config parameters."""
    from instructor.providers.gemini.utils import handle_genai_structured_outputs
    
    test_labels = {"environment": "test"}
    
    kwargs_with_mixed_config = {
        "messages": [{"role": "user", "content": "Test"}],
        "config": {
            "labels": test_labels,
            "system_instruction": "You are a helpful assistant"
        },
        "generation_config": {
            "temperature": 0.7,
            "max_tokens": 100
        }
    }
    
    result_model, result_kwargs = handle_genai_structured_outputs(
        SimpleResponse, kwargs_with_mixed_config.copy()
    )
    
    config = result_kwargs["config"]
    assert hasattr(config, 'labels'), "Config should have labels attribute"
    assert config.labels == test_labels, "Labels should be preserved"


def test_config_without_labels():
    """Test that the fix doesn't break when no labels are provided."""
    from instructor.providers.gemini.utils import handle_genai_structured_outputs
    
    kwargs_without_labels = {
        "messages": [{"role": "user", "content": "Test"}],
        "config": {"system_instruction": "You are helpful"}
    }
    
    result_model, result_kwargs = handle_genai_structured_outputs(
        SimpleResponse, kwargs_without_labels.copy()
    )
    
    config = result_kwargs["config"]
    assert hasattr(config, 'system_instruction')
    
    if hasattr(config, 'labels'):
        assert config.labels is None or config.labels == {}


def test_no_config_provided():
    """Test that the fix doesn't break when no config is provided at all."""
    from instructor.providers.gemini.utils import handle_genai_structured_outputs
    
    kwargs_no_config = {
        "messages": [{"role": "user", "content": "Test"}]
    }
    
    result_model, result_kwargs = handle_genai_structured_outputs(
        SimpleResponse, kwargs_no_config.copy()
    )
    
    assert "config" in result_kwargs
    config = result_kwargs["config"]
    
    if hasattr(config, 'labels'):
        assert config.labels is None or config.labels == {}


def test_update_genai_kwargs_preserves_labels():
    """Test that update_genai_kwargs preserves labels from config."""
    from instructor.providers.gemini.utils import update_genai_kwargs
    
    test_labels = {"environment": "test", "user_id": "123"}
    
    kwargs = {
        "config": {"labels": test_labels},
        "generation_config": {"temperature": 0.7}
    }
    base_config = {}
    
    result = update_genai_kwargs(kwargs, base_config)
    
    assert "labels" in result
    assert result["labels"] == test_labels


def test_update_genai_kwargs_no_labels():
    """Test that update_genai_kwargs works when no labels are provided."""
    from instructor.providers.gemini.utils import update_genai_kwargs
    
    kwargs = {
        "config": {"system_instruction": "You are helpful"},
        "generation_config": {"temperature": 0.7}
    }
    base_config = {}
    
    result = update_genai_kwargs(kwargs, base_config)
    
    assert "labels" not in result
    assert result["temperature"] == 0.7


def test_update_genai_kwargs_empty_config():
    """Test that update_genai_kwargs works with empty or missing config."""
    from instructor.providers.gemini.utils import update_genai_kwargs
    
    kwargs = {
        "generation_config": {"temperature": 0.7}
    }
    base_config = {}
    
    result = update_genai_kwargs(kwargs, base_config)
    
    assert "labels" not in result
    assert result["temperature"] == 0.7
    
    kwargs_empty_config = {
        "config": {},
        "generation_config": {"temperature": 0.5}
    }
    
    result = update_genai_kwargs(kwargs_empty_config, {})
    
    assert "labels" not in result
    assert result["temperature"] == 0.5
