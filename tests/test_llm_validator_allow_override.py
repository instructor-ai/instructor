"""Tests for llm_validator allow_override functionality.

Verifies that the allow_override parameter in llm_validator correctly
returns a fixed value when the LLM deems the input invalid, instead of
raising an AssertionError.
"""

from unittest.mock import Mock

import pytest

from instructor.processing.validators import Validator
from instructor.validation.llm_validators import llm_validator


def _make_mock_client(
    *, is_valid: bool, reason: str | None = None, fixed_value: str | None = None
):
    """Create a mock instructor client that returns a predetermined Validator response."""
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = Validator(
        is_valid=is_valid,
        reason=reason,
        fixed_value=fixed_value,
    )
    return mock_client


class TestAllowOverride:
    """Tests for the allow_override parameter in llm_validator."""

    def test_valid_value_returns_original(self):
        """When the LLM deems the value valid, the original value is returned."""
        client = _make_mock_client(is_valid=True)
        validator = llm_validator(
            statement="Must be lowercase",
            client=client,
            allow_override=False,
        )

        result = validator("jason liu")
        assert result == "jason liu"

    def test_invalid_without_override_raises(self):
        """When the value is invalid and allow_override is False, an AssertionError is raised."""
        client = _make_mock_client(
            is_valid=False,
            reason="Name is not lowercase",
            fixed_value="jason liu",
        )
        validator = llm_validator(
            statement="Must be lowercase",
            client=client,
            allow_override=False,
        )

        with pytest.raises(AssertionError, match="Name is not lowercase"):
            validator("Jason Liu")

    def test_invalid_with_override_returns_fixed_value(self):
        """When allow_override is True and the LLM provides a fixed value, that value is returned."""
        client = _make_mock_client(
            is_valid=False,
            reason="Name is not lowercase",
            fixed_value="jason liu",
        )
        validator = llm_validator(
            statement="Must be lowercase",
            client=client,
            allow_override=True,
        )

        result = validator("Jason Liu")
        assert result == "jason liu"

    def test_invalid_with_override_but_no_fixed_value_raises(self):
        """When allow_override is True but the LLM provides no fixed value, an AssertionError is raised."""
        client = _make_mock_client(
            is_valid=False,
            reason="Name is not lowercase",
            fixed_value=None,
        )
        validator = llm_validator(
            statement="Must be lowercase",
            client=client,
            allow_override=True,
        )

        with pytest.raises(AssertionError, match="Name is not lowercase"):
            validator("Jason Liu")

    def test_valid_value_with_override_returns_original(self):
        """When the value is valid, allow_override has no effect and the original is returned."""
        client = _make_mock_client(is_valid=True)
        validator = llm_validator(
            statement="Must be lowercase",
            client=client,
            allow_override=True,
        )

        result = validator("jason liu")
        assert result == "jason liu"
