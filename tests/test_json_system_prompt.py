"""
Tests for json_system_prompt customization feature.

Tests verify that:
- Users can customize the JSON mode system prompt
- {schema} placeholder is correctly substituted
- Empty string disables system prompt modification
- Default behavior is backward compatible
"""

import json
import pytest
from unittest.mock import MagicMock

from instructor.providers.openai.utils import handle_json_modes
from instructor.mode import Mode
from pydantic import BaseModel


class SimpleModel(BaseModel):
    """Test model for JSON schema generation."""
    name: str
    age: int


class TestJsonSystemPromptCustomization:
    """Tests for Issue #1514 - Customizable JSON mode system prompt."""

    def test_default_prompt_backward_compatible(self):
        """Default behavior should be unchanged (backward compatible)."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Extract data"}]
        }
        
        response_model, result_kwargs = handle_json_modes(
            SimpleModel, new_kwargs, Mode.JSON
        )
        
        # Should have inserted system message
        assert result_kwargs["messages"][0]["role"] == "system"
        # Should contain the default "genius expert" phrase
        assert "genius expert" in result_kwargs["messages"][0]["content"]
        # Should contain the JSON schema
        assert "SimpleModel" in result_kwargs["messages"][0]["content"] or \
               "name" in result_kwargs["messages"][0]["content"]

    def test_custom_prompt_with_schema_placeholder(self):
        """Custom prompt with {schema} placeholder should work."""
        custom_prompt = "You are a helpful assistant. Return JSON matching:\n{schema}"
        new_kwargs = {
            "messages": [{"role": "user", "content": "Extract data"}]
        }
        
        response_model, result_kwargs = handle_json_modes(
            SimpleModel, new_kwargs, Mode.JSON, json_system_prompt=custom_prompt
        )
        
        # Should have inserted system message
        assert result_kwargs["messages"][0]["role"] == "system"
        # Should NOT contain the default phrase
        assert "genius expert" not in result_kwargs["messages"][0]["content"]
        # Should contain our custom text
        assert "helpful assistant" in result_kwargs["messages"][0]["content"]
        # Schema should be substituted
        assert "name" in result_kwargs["messages"][0]["content"]

    def test_empty_string_skips_system_prompt(self):
        """Empty string should skip system prompt modification entirely."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Extract data"}]
        }
        
        response_model, result_kwargs = handle_json_modes(
            SimpleModel, new_kwargs, Mode.JSON, json_system_prompt=""
        )
        
        # Should NOT have inserted system message
        assert result_kwargs["messages"][0]["role"] == "user"
        # Original message should be unchanged
        assert result_kwargs["messages"][0]["content"] == "Extract data"

    def test_custom_prompt_appends_to_existing_system(self):
        """Custom prompt should append to existing system message."""
        custom_prompt = "Respond with JSON: {schema}"
        new_kwargs = {
            "messages": [
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Tell me about treasure"}
            ]
        }
        
        response_model, result_kwargs = handle_json_modes(
            SimpleModel, new_kwargs, Mode.JSON, json_system_prompt=custom_prompt
        )
        
        # System message should be preserved and extended
        assert result_kwargs["messages"][0]["role"] == "system"
        assert "pirate" in result_kwargs["messages"][0]["content"]
        assert "JSON" in result_kwargs["messages"][0]["content"]

    def test_json_schema_mode_ignores_system_prompt(self):
        """JSON_SCHEMA mode uses response_format, not system prompt modification."""
        custom_prompt = "Custom prompt {schema}"
        new_kwargs = {
            "messages": [{"role": "user", "content": "Extract data"}]
        }
        
        response_model, result_kwargs = handle_json_modes(
            SimpleModel, new_kwargs, Mode.JSON_SCHEMA, json_system_prompt=custom_prompt
        )
        
        # Should have response_format set
        assert "response_format" in result_kwargs
        assert result_kwargs["response_format"]["type"] == "json_schema"
        # System message should NOT be inserted for JSON_SCHEMA mode
        assert result_kwargs["messages"][0]["role"] == "user"

    def test_md_json_mode_with_custom_prompt(self):
        """MD_JSON mode should work with custom prompt."""
        custom_prompt = "Return markdown JSON: {schema}"
        new_kwargs = {
            "messages": [{"role": "user", "content": "Extract data"}]
        }
        
        response_model, result_kwargs = handle_json_modes(
            SimpleModel, new_kwargs, Mode.MD_JSON, json_system_prompt=custom_prompt
        )
        
        # Should have system message with custom prompt
        assert result_kwargs["messages"][0]["role"] == "system"
        assert "markdown JSON" in result_kwargs["messages"][0]["content"]

    def test_none_response_model_returns_early(self):
        """None response_model should return early without modification."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        response_model, result_kwargs = handle_json_modes(
            None, new_kwargs, Mode.JSON, json_system_prompt="custom"
        )
        
        assert response_model is None
        # Messages should be unchanged
        assert result_kwargs["messages"][0]["role"] == "user"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
