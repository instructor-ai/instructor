"""
Tests for llm_validator allow_override functionality.

Tests cover:
- Valid input returns unchanged
- Invalid input with allow_override=True returns fixed_value
- Invalid input with allow_override=False raises ValueError
- Invalid input without fixed_value raises ValueError
"""

import pytest
from unittest.mock import MagicMock, patch

from instructor.validation.llm_validators import llm_validator
from instructor.processing.validators import Validator


class TestLLMValidatorAllowOverride:
    """Tests for the allow_override parameter fix."""

    def test_valid_input_returns_unchanged(self):
        """Test that valid input is returned unchanged."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = Validator(
            is_valid=True,
            reason=None,
            fixed_value=None,
        )

        validator = llm_validator(
            statement="must be lowercase",
            client=mock_client,
            allow_override=False,
        )

        result = validator("hello")
        assert result == "hello"

    def test_invalid_with_override_returns_fixed_value(self):
        """Test that invalid input with allow_override=True returns fixed_value."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = Validator(
            is_valid=False,
            reason="Name should be lowercase",
            fixed_value="jason liu",
        )

        validator = llm_validator(
            statement="must be lowercase",
            client=mock_client,
            allow_override=True,
        )

        result = validator("Jason Liu")
        assert result == "jason liu"

    def test_invalid_without_override_raises_value_error(self):
        """Test that invalid input with allow_override=False raises ValueError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = Validator(
            is_valid=False,
            reason="Name should be lowercase",
            fixed_value="jason liu",
        )

        validator = llm_validator(
            statement="must be lowercase",
            client=mock_client,
            allow_override=False,
        )

        with pytest.raises(ValueError, match="Name should be lowercase"):
            validator("Jason Liu")

    def test_invalid_with_override_but_no_fixed_value_raises(self):
        """Test that invalid input without fixed_value raises ValueError even with override."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = Validator(
            is_valid=False,
            reason="Invalid input, cannot fix",
            fixed_value=None,
        )

        validator = llm_validator(
            statement="must be a valid email",
            client=mock_client,
            allow_override=True,
        )

        with pytest.raises(ValueError, match="Invalid input, cannot fix"):
            validator("not-an-email")

    def test_default_allow_override_is_false(self):
        """Test that default behavior (allow_override=False) raises on invalid."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = Validator(
            is_valid=False,
            reason="Validation failed",
            fixed_value="fixed",
        )

        # Don't pass allow_override, use default
        validator = llm_validator(
            statement="must be valid",
            client=mock_client,
        )

        # Default should be False, so it should raise
        with pytest.raises(ValueError):
            validator("invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
