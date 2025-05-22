"""Example test file demonstrating unified test infrastructure.

This shows how the new infrastructure reduces code duplication
and makes testing consistent across providers.
"""

import pytest
from .base import BaseProviderTest
from .test_models import ValidationError


class TestUnifiedInfrastructure:
    """Test the unified test infrastructure itself."""

    def test_common_models_available(self):
        """Test that common models are importable."""
        # These would have been duplicated across every provider before
        from .test_models import (
            UserExtract,
            Labels,
            CLASSIFICATION_TEST_DATA,
        )

        # Create instances to verify they work
        user = UserExtract(name="Test", age=25)
        assert user.name == "Test"
        assert user.age == 25

        # Test data is available
        assert len(CLASSIFICATION_TEST_DATA) > 0
        assert CLASSIFICATION_TEST_DATA[0][1] == Labels.SPAM

    def test_enhanced_exceptions(self):
        """Test enhanced exception features."""
        # Test ValidationError with guidance
        error = ValidationError(
            "Validation failed",
            validation_errors=[
                {"loc": ["name"], "type": "missing", "msg": "Field required"},
                {"loc": ["age"], "type": "type_error", "msg": "Not a valid integer"},
            ],
            model_name="UserExtract",
        )

        # Check guidance is generated
        assert "Field 'name' is required" in error.guidance
        assert "Field 'age' has wrong type" in error.guidance

        # Check error details
        assert error.model_name == "UserExtract"
        assert len(error.validation_errors) == 2

        # Check structured logging capability
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "ValidationError"
        assert "validation_errors" in error_dict["details"]
        assert error_dict["guidance"] is not None

    def test_base_provider_test_skip_logic(self):
        """Test that base test class skip logic works."""

        class MockProviderTest(BaseProviderTest):
            @property
            def provider_config(self):
                from .base import ProviderConfig

                return ProviderConfig(
                    name="mock",
                    models=["model-1"],
                    modes=[],
                    client_factory=lambda c, m: c,  # noqa: ARG005
                    env_var="MOCK_API_KEY",
                )

        test_instance = MockProviderTest()

        # Without API key, should skip
        import os

        original_value = os.environ.get("MOCK_API_KEY")
        try:
            if "MOCK_API_KEY" in os.environ:
                del os.environ["MOCK_API_KEY"]

            with pytest.raises(pytest.skip.Exception):
                test_instance.skip_if_no_api_key()
        finally:
            if original_value:
                os.environ["MOCK_API_KEY"] = original_value


# Example of how much simpler provider tests become:
"""
# BEFORE: Each provider had ~500 lines of duplicated test code
# test_openai/test_simple.py, test_anthropic/test_simple.py, etc.

# AFTER: Just define config and get all tests automatically
from .base import ProviderConfig, create_provider_test_class

config = ProviderConfig(
    name="newprovider",
    models=["model-1"],
    modes=[Mode.JSON],
    client_factory=lambda c, m: instructor.from_newprovider(c, mode=m),
)

TestNewProvider = create_provider_test_class(config)
# That's it! All common tests are automatically included
"""
