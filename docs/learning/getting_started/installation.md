# Instructor Installation Guide: Setup for LLM Structured Outputs

Learn how to install Instructor, the leading Python library for extracting structured data from LLMs like GPT-4, Claude, and Gemini. This comprehensive installation tutorial covers all major LLM providers and gets you ready for production use.

## Quick Start: Install Instructor for LLM Development

Get started with structured LLM outputs in seconds. Install Instructor with Google GenAI support using `uv`:

```shell
uv add "instructor[google-genai]"
```

This installs Instructor together with the Google GenAI SDK, which we recommend as the default path for new projects.

## LLM Provider Installation Guide

Instructor supports 15+ LLM providers. Here's how to install and configure each:

### Google GenAI (Gemini Models)

Google's Gemini models provide fast multimodal function calling and structured outputs. This is our recommended starting point:

```shell
uv add "instructor[google-genai]"
```

Configure your Google API key:

```shell
export GOOGLE_API_KEY=your_google_key
```

### Anthropic Claude LLM Setup

Install the Anthropic extra only when you need Claude models:

```shell
uv add "instructor[anthropic]"
```

Configure Claude API access:

```shell
export ANTHROPIC_API_KEY=your_anthropic_key
```

### Cohere

To use with Cohere's models:

```shell
uv add "instructor[cohere]"
```

Set up your Cohere API key:

```shell
export COHERE_API_KEY=your_cohere_key
```

### Mistral

To use with Mistral AI's models:

```shell
uv add "instructor[mistralai]"
```

Set up your Mistral API key:

```shell
export MISTRAL_API_KEY=your_mistral_key
```

### LiteLLM (Multiple Providers)

To use LiteLLM for accessing multiple providers:

```shell
uv add "instructor[litellm]"
```

Set up API keys for the providers you want to use.

## Verify Your Instructor LLM Setup

Test your Instructor installation with this simple LLM structured output example:

```python
import asyncio
import instructor
from instructor import Mode
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


async def verify_install() -> None:
    client = instructor.from_provider(
        "google/gemini-2.5-flash",
        async_client=True,
        mode=Mode.GENAI_STRUCTURED_OUTPUTS,
    )

    person = await client.chat.completions.create(
        response_model=Person,
        messages=[
            {"role": "user", "content": "John Doe is 30 years old"}
        ],
    )

    print(f"Name: {person.name}, Age: {person.age}")


asyncio.run(verify_install())
```

## Next Steps in Your LLM Tutorial Journey

With Instructor installed, you're ready to build powerful LLM applications:

1. **[Create Your First LLM Extraction](first_extraction.md)** - Build structured outputs with any LLM
2. **[Master Response Models](response_models.md)** - Learn Pydantic models for LLM data validation
3. **[Configure LLM Clients](client_setup.md)** - Set up OpenAI, Anthropic, Google, and more

## Common Installation Issues

- **Import Errors**: Ensure you've added the provider-specific extras (e.g., `uv add "instructor[anthropic]"`)
- **API Key Issues**: Verify your environment variables are set correctly
- **Version Conflicts**: Use `uv pip install --upgrade "instructor[google-genai]"` to get the latest version

Ready to extract structured data from LLMs? Continue to [Your First Extraction](first_extraction.md) →