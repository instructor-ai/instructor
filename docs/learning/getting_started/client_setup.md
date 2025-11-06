# Client Setup

Setting up the right client is the first step in using Instructor with various LLM providers. This guide covers how to configure clients for different providers and explains the various modes available.

## Google GenAI (Recommended)

Start with Google's Gemini models using the async client:

```python
import asyncio
import instructor
from instructor import Mode


async def main() -> None:
    client = instructor.from_provider(
        "google/gemini-2.5-flash",
        async_client=True,
        mode=Mode.GENAI_STRUCTURED_OUTPUTS,
    )

    result = await client.chat.completions.create(
        response_model=dict,
        messages=[{"role": "user", "content": "Extract the topic and tone."}],
    )

    print(result)


asyncio.run(main())
```

Set `vertexai=True` when you need to run against Google Cloud Vertex AI.

## OpenAI

OpenAI remains fully supported:

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(
    OpenAI(),
    mode=instructor.Mode.TOOLS,
)
```

## Anthropic (Claude)

For Anthropic's Claude models:

```python
import instructor

client = instructor.from_provider(
    "anthropic/claude-3-5-haiku-latest",
    mode=instructor.Mode.ANTHROPIC_TOOLS,
)
```

## Cohere

For Cohere's models:

```python
import instructor
import cohere

cohere_client = cohere.ClientV2("YOUR_API_KEY")
client = instructor.from_cohere(cohere_client)
```

## Mistral

For Mistral AI's models:

```python
import instructor
from mistralai.client import MistralClient

mistral_client = MistralClient(api_key="YOUR_API_KEY")
client = instructor.from_mistral(mistral_client)
```

## Understanding Modes

Instructor supports different modes for structured extraction:

```python
from instructor import Mode

Mode.GENAI_STRUCTURED_OUTPUTS  # Preferred for Google GenAI JSON-style responses
Mode.GENAI_TOOLS               # Google function calling format
Mode.TOOLS                     # OpenAI function calling format
Mode.JSON                      # Plain JSON generation
Mode.MD_JSON                   # Markdown JSON (used by some providers)
Mode.ANTHROPIC_TOOLS           # Claude tools mode (default for Anthropic)
Mode.GEMINI_TOOLS              # Legacy Gemini tools format
Mode.GEMINI_JSON               # Legacy Gemini JSON format
```

### When to Use Each Mode

- **GENAI_STRUCTURED_OUTPUTS**: Best default for Google GenAI. Returns JSON that maps straight to your model.
- **GENAI_TOOLS**: Use when you need Google function calling or tool definitions.
- **TOOLS/FUNCTION_CALL**: Default for OpenAI. Uses function calling for reliable structured outputs.
- **JSON**: Works with most providers. The model generates JSON directly.
- **MD_JSON**: For models that work well with Markdown-formatted JSON.
- **ANTHROPIC_TOOLS**: Default for Anthropic. Uses Claude's tools API.
- **GEMINI_TOOLS/GEMINI_JSON**: Legacy Gemini modes. Prefer the GenAI modes for new builds.

Choose the mode that works best with your selected provider and model.

## Async Clients

For asynchronous operation, use the async versions of the clients:

```python
import asyncio
import instructor
from instructor import Mode
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


async def extract_user():
    async_client = instructor.from_provider(
        "google/gemini-2.5-flash",
        async_client=True,
        mode=Mode.GENAI_STRUCTURED_OUTPUTS,
    )
    return await async_client.chat.completions.create(
        response_model=User,
        messages=[
            {"role": "user", "content": "John is 30 years old."}
        ],
    )


user = asyncio.run(extract_user())
```

Similar async patterns work for other providers like Anthropic and Mistral.

## Advanced OpenAI Configuration

You can pass additional parameters to the OpenAI client:

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(
    OpenAI(),
    mode=instructor.Mode.TOOLS,
    max_retries=2,
)

# With API key explicitly defined
client = instructor.from_provider("openai/gpt-5-nano", mode=instructor.Mode.JSON)
```

## Using with Other Providers via OpenAI-Compatible Interface

Many providers offer an OpenAI-compatible API:

```python
import instructor
# Example for Azure OpenAI
azure_client = instructor.from_provider("openai/gpt-5-nano")

# Example for Groq
groq_client = instructor.from_provider("openai/gpt-5-nano")
```

## Next Steps

Now that you've set up your client, you can:

1. Create [simple object extractions](../patterns/simple_object.md)
2. Work with [lists](../patterns/list_extraction.md) and [nested structures](../patterns/nested_structure.md)
3. Add [validation](../validation/basics.md) to your models
4. Handle [optional fields](../patterns/optional_fields.md)

The following sections will guide you through these patterns with increasingly complex examples.
