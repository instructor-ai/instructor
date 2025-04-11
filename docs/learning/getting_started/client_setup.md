# Client Setup

Setting up the right client is the first step in using Instructor with various LLM providers. This guide covers how to configure clients for different providers and explains the various modes available.

## OpenAI

The most common configuration is with OpenAI:

```python
import instructor
from openai import OpenAI

# Default mode (TOOLS)
client = instructor.from_openai(OpenAI())

# With JSON mode
client = instructor.from_openai(
    OpenAI(),
    mode=instructor.Mode.JSON  # Use JSON mode instead
)
```

## Anthropic (Claude)

For Anthropic's Claude models:

```python
import instructor
from anthropic import Anthropic

# Default mode (ANTHROPIC_TOOLS)
client = instructor.from_anthropic(Anthropic())

# With JSON mode
client = instructor.from_anthropic(
    Anthropic(),
    mode=instructor.Mode.JSON
)
```

## Google Gemini

For Google's Gemini models:

```python
import instructor
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

client = instructor.from_gemini(
    model,
    mode=instructor.Mode.GEMINI_TOOLS  # or GEMINI_JSON
)
```

## Cohere

For Cohere's models:

```python
import instructor
import cohere

cohere_client = cohere.Client("YOUR_API_KEY")
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

Mode.TOOLS         # OpenAI function calling format (default for OpenAI)
Mode.JSON          # Plain JSON generation
Mode.MD_JSON       # Markdown JSON (used by some providers)
Mode.ANTHROPIC_TOOLS # Claude tools mode (default for Anthropic)
Mode.GEMINI_TOOLS  # Gemini tools format
Mode.GEMINI_JSON   # Gemini JSON format
```

### When to Use Each Mode

- **TOOLS/FUNCTION_CALL**: The default for OpenAI. Uses function calling for reliable structured outputs.
- **JSON**: Works with most providers. The model generates JSON directly.
- **MD_JSON**: For models that work well with Markdown-formatted JSON.
- **ANTHROPIC_TOOLS**: The default for Anthropic. Uses Claude's tools API.
- **GEMINI_TOOLS/GEMINI_JSON**: For Google's Gemini models.

Choose the mode that works best with your selected provider and model.

## Async Clients

For asynchronous operation, use the async versions of the clients:

```python
import asyncio
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    async_client = instructor.from_openai(AsyncOpenAI())
    return await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=User,
        messages=[
            {"role": "user", "content": "John is 30 years old."}
        ]
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
    max_retries=2,  # Number of retries for validation failures
)

# With API key explicitly defined
client = instructor.from_openai(
    OpenAI(api_key="your-api-key"),
    mode=instructor.Mode.JSON
)

# With organization ID
client = instructor.from_openai(
    OpenAI(
        api_key="your-api-key",
        organization="org-..."
    )
)
```

## Using with Other Providers via OpenAI-Compatible Interface

Many providers offer an OpenAI-compatible API:

```python
import instructor
from openai import OpenAI

# Example for Azure OpenAI
azure_client = instructor.from_openai(
    OpenAI(
        api_key="your-azure-api-key",
        base_url="https://your-resource-name.openai.azure.com/openai/deployments/your-deployment-name"
    )
)

# Example for Groq
groq_client = instructor.from_openai(
    OpenAI(
        api_key="your-groq-api-key",
        base_url="https://api.groq.com/openai/v1"
    )
)
```

## Next Steps

Now that you've set up your client, you can:

1. Create [simple object extractions](../patterns/simple_object.md)
2. Work with [lists](../patterns/list_extraction.md) and [nested structures](../patterns/nested_structure.md)
3. Add [validation](../validation/basics.md) to your models
4. Handle [optional fields](../patterns/optional_fields.md)

The following sections will guide you through these patterns with increasingly complex examples. 