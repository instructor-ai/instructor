"""Example of using unified test infrastructure for OpenAI.

This demonstrates how the new test infrastructure reduces duplication
and provides consistent testing across providers.
"""

from openai import OpenAI, AsyncOpenAI
import instructor
from instructor.mode import Mode
from .base import ProviderConfig, BaseClientFixtures, create_provider_test_class


# Define OpenAI configuration
OPENAI_CONFIG = ProviderConfig(
    name="openai",
    models=["gpt-4o-mini", "gpt-4o"],
    modes=[
        Mode.TOOLS,
        Mode.TOOLS_STRICT,
        Mode.JSON,
        Mode.MD_JSON,
        Mode.PARALLEL_TOOLS,
    ],
    client_factory=lambda client, mode: instructor.from_openai(client, mode=mode),
    supports_async=True,
    supports_streaming=True,
    supports_tools=True,
    supports_json=True,
    env_var="OPENAI_API_KEY",
    skip_models=["gpt-4o"] if os.environ.get("LIMIT_PROVIDER_MODELS") else None,
)

# Create client fixtures
client, async_client = BaseClientFixtures.create_client_fixture(
    ClientClass=OpenAI,
    AsyncClientClass=AsyncOpenAI,
    api_key_env="OPENAI_API_KEY",
)


# Create test class using factory
TestOpenAI = create_provider_test_class(OPENAI_CONFIG)


# You can also extend the test class for provider-specific tests
class TestOpenAISpecific(TestOpenAI):
    """OpenAI-specific tests that don't apply to other providers."""

    def test_parallel_tools(self, client):
        """Test OpenAI's parallel tool calling feature."""
        self.skip_if_no_api_key()

        from .test_models import UserExtract

        instructor_client = instructor.from_openai(client, mode=Mode.PARALLEL_TOOLS)

        # Test parallel extraction
        response = instructor_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Extract all users: Jason (25), Sarah (30), Mike (28)",
                }
            ],
            response_model=list[UserExtract],
        )

        assert len(response) == 3
        assert all(isinstance(u, UserExtract) for u in response)

    def test_json_mode_with_schema(self, client):
        """Test JSON mode with response_format schema (OpenAI specific)."""
        self.skip_if_no_api_key()

        from .test_models import UserExtract

        instructor_client = instructor.from_openai(client, mode=Mode.TOOLS_STRICT)

        response = instructor_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
            response_model=UserExtract,
        )

        assert response.name == "Jason"
        assert response.age == 25


# Example of how simple it is to add tests for a new provider
"""
# For a new provider, you would just need:

from .base import ProviderConfig

NEW_PROVIDER_CONFIG = ProviderConfig(
    name="newprovider",
    models=["model-1", "model-2"],
    modes=[Mode.JSON],  # Whatever modes are supported
    client_factory=lambda c, m: instructor.from_newprovider(c, mode=m),
    env_var="NEW_PROVIDER_API_KEY",
)

# Create fixtures
client, async_client = BaseClientFixtures.create_client_fixture(
    ClientClass=NewProviderClient,
    api_key_env="NEW_PROVIDER_API_KEY",
)

# Create test class - automatically gets all common tests!
TestNewProvider = create_provider_test_class(NEW_PROVIDER_CONFIG)
"""
