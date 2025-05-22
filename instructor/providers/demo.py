"""
Demo script showing the new provider registry architecture.

This demonstrates how the refactored system works with minimal changes
to the existing codebase while providing much better extensibility.
"""

from pydantic import BaseModel

from instructor.mode import Mode
from instructor.providers import ProviderRegistry


class User(BaseModel):
    name: str
    age: int


def demonstrate_provider_registry():
    """Show how the provider registry works."""

    print("=== Provider Registry Demo ===\n")

    # List all registered providers
    print("Registered providers:")
    for provider_name in ProviderRegistry.list_providers():
        provider = ProviderRegistry.get_provider(provider_name)
        print(f"- {provider_name}: supports modes {provider.get_supported_modes()}")

    print("\n--- Mode to Provider Mapping ---")

    # Show which provider handles which mode
    all_modes = [
        Mode.TOOLS,
        Mode.JSON,
        Mode.ANTHROPIC_TOOLS,
        Mode.GEMINI_JSON,
        Mode.MISTRAL_TOOLS,
        Mode.COHERE_TOOLS,
    ]

    for mode in all_modes:
        provider = ProviderRegistry.get_provider_for_mode(mode)
        if provider:
            print(f"{mode.value} -> {provider.name}")
        else:
            print(f"{mode.value} -> No provider registered")

    print("\n--- Request Preparation Example ---")

    # Example: Prepare an OpenAI tools request
    openai_provider = ProviderRegistry.get_provider("openai")
    if openai_provider and hasattr(openai_provider, "prepare_request"):
        model, kwargs = openai_provider.prepare_request(
            User,
            {"messages": [{"role": "user", "content": "Extract user info"}]},
            Mode.TOOLS,
        )
        print(f"OpenAI TOOLS mode preparation:")
        print(f"  - Model: {model}")
        print(f"  - Added 'tools' to kwargs: {'tools' in kwargs}")
        print(f"  - Added 'tool_choice' to kwargs: {'tool_choice' in kwargs}")

    print("\n--- Adding a Custom Provider ---")

    # Show how easy it is to add a new provider
    from instructor.providers import BaseProvider

    @ProviderRegistry.register("custom")
    class CustomProvider(BaseProvider):
        @property
        def name(self):
            return "custom"

        def get_supported_modes(self):
            return {Mode.JSON}  # Just support JSON mode

        def validate_response(self, response, mode):  # noqa: ARG002
            pass

        def process_response(self, response, response_model, mode, **kwargs):  # noqa: ARG002
            # Custom processing logic here
            return response_model.model_validate_json(response)

        async def process_response_async(
            self, response, response_model, mode, **kwargs
        ):
            return self.process_response(response, response_model, mode, **kwargs)

        def create_instructor(self, client, mode, **kwargs):  # noqa: ARG002
            from instructor.patch import patch

            return patch(client, mode=mode)

    print("Added custom provider!")
    custom = ProviderRegistry.get_provider("custom")
    print(f"Custom provider supports: {custom.get_supported_modes()}")

    print("\n=== Benefits ===")
    print("1. No more 1000+ line process_response.py file")
    print("2. Each provider manages its own logic")
    print("3. Easy to add new providers without touching core code")
    print("4. Better testability - test providers in isolation")
    print("5. Clear separation of concerns")


if __name__ == "__main__":
    # Import the providers to register them

    demonstrate_provider_registry()
