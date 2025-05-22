"""Common feature tests that run across all providers.

This module contains parametrized tests that validate common functionality
across all LLM providers, ensuring consistent behavior.
"""

from .base import BaseProviderTest, skip_if_missing_features
from .test_models import (
    UserExtract,
    UserDetail,
    Order,
    SinglePrediction,
    PartialUser,
    CLASSIFICATION_TEST_DATA,
    USER_EXTRACTION_PROMPTS,
    NESTED_EXTRACTION_PROMPT,
)


class CommonProviderTests(BaseProviderTest):
    """Test suite with common tests for all providers.

    Provider-specific test classes should inherit from this class
    and implement the provider_config property.
    """

    def test_basic_extraction(self, client):
        """Test basic model extraction."""
        self.skip_if_no_api_key()

        for model in self.get_test_models():
            for mode in self.get_test_modes():
                instructor_client = self.get_client(client, mode)

                response = instructor_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": USER_EXTRACTION_PROMPTS[0]}],
                    response_model=UserExtract,
                )

                assert isinstance(response, UserExtract)
                assert response.name == "Jason"
                assert response.age == 25

    def test_validation_retry(self, client):
        """Test validation with retry logic."""
        self.skip_if_no_api_key()

        for model in self.get_test_models():
            for mode in self.get_test_modes():
                instructor_client = self.get_client(client, mode)

                response = instructor_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Extract: jason is 25 years old",
                        }  # lowercase
                    ],
                    response_model=UserDetail,
                    max_retries=3,
                )

                assert isinstance(response, UserDetail)
                assert response.name == "JASON"  # Should be uppercase after retry
                assert response.age == 25

    def test_nested_models(self, client):
        """Test extraction with nested models."""
        self.skip_if_no_api_key()

        for model in self.get_test_models():
            for mode in self.get_test_modes():
                instructor_client = self.get_client(client, mode)

                response = instructor_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": NESTED_EXTRACTION_PROMPT}],
                    response_model=Order,
                )

                assert isinstance(response, Order)
                assert response.order_id == "ORD-001"
                assert response.customer_name == "John Doe"
                assert len(response.items) == 2
                assert response.total == 45.0  # Calculated by validator

    @skip_if_missing_features("streaming")
    def test_iterable_extraction(self, client):
        """Test extraction of multiple items using iterable."""
        self.skip_if_no_api_key()

        for model in self.get_test_models():
            for mode in self.get_test_modes():
                instructor_client = self.get_client(client, mode)

                users = instructor_client.chat.completions.create_iterable(
                    model=model,
                    messages=[
                        {"role": "user", "content": "\n".join(USER_EXTRACTION_PROMPTS)}
                    ],
                    response_model=UserExtract,
                )

                extracted_users = list(users)
                assert len(extracted_users) == 3
                assert extracted_users[0].name == "Jason"
                assert extracted_users[1].name == "Sarah"
                assert extracted_users[2].name == "Mike"

    @skip_if_missing_features("streaming")
    def test_partial_streaming(self, client):
        """Test partial/streaming responses."""
        self.skip_if_no_api_key()

        for model in self.get_test_models():
            for mode in self.get_test_modes():
                instructor_client = self.get_client(client, mode)

                stream = instructor_client.chat.completions.create_partial(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Create a user: Jason, 25, software engineer",
                        }
                    ],
                    response_model=PartialUser,
                )

                partials = list(stream)
                assert len(partials) > 1  # Should have multiple partial updates

                # First partial might have incomplete data
                assert partials[0].name is not None or partials[0].age is not None

                # Final partial should be complete
                final = partials[-1]
                assert final.name == "Jason"
                assert final.age == 25
                assert "engineer" in final.bio.lower()

    def test_classification(self, client):
        """Test enum-based classification."""
        self.skip_if_no_api_key()

        for model in self.get_test_models():
            for mode in self.get_test_modes():
                instructor_client = self.get_client(client, mode)

                # Test a few classification examples
                for text, expected_label in CLASSIFICATION_TEST_DATA[:2]:
                    response = instructor_client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Classify as spam or not: {text}",
                            }
                        ],
                        response_model=SinglePrediction,
                    )

                    assert isinstance(response, SinglePrediction)
                    assert response.class_label == expected_label

    def test_empty_response_handling(self, client):  # noqa: ARG002
        """Test handling of empty or None responses."""
        self.skip_if_no_api_key()

        # This test might need provider-specific handling
        # as different providers handle edge cases differently
        pass

    @skip_if_missing_features("async")
    async def test_async_extraction(self, async_client):
        """Test async client functionality."""
        self.skip_if_no_api_key()

        for model in self.get_test_models():
            for mode in self.get_test_modes():
                instructor_client = self.get_client(async_client, mode)

                response = await instructor_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": USER_EXTRACTION_PROMPTS[0]}],
                    response_model=UserExtract,
                )

                assert isinstance(response, UserExtract)
                assert response.name == "Jason"
                assert response.age == 25


def create_provider_test_class(provider_config):
    """Factory function to create a test class for a specific provider.

    Args:
        provider_config: ProviderConfig instance

    Returns:
        Test class configured for the provider
    """

    class ProviderTests(CommonProviderTests):
        @property
        def provider_config(self):
            return provider_config

    # Set class name for better test reporting
    ProviderTests.__name__ = f"Test{provider_config.name.title()}"

    return ProviderTests
