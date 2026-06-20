---
title: Structured Outputs with Azure OpenAI and AI Foundry
description: Learn how to use Azure OpenAI and Azure AI Foundry with Instructor for structured JSON outputs using Pydantic models.
---

# Structured Outputs with Azure OpenAI / AI Foundry

This guide demonstrates how to use Azure OpenAI (and Azure AI Foundry-hosted
models) with Instructor to generate structured outputs. Azure OpenAI provides
an OpenAI-compatible API, so all standard Instructor modes are supported.

## Prerequisites

Create an Azure OpenAI resource and deploy a model in the [Azure Portal](https://portal.azure.com).
Set the following environment variables:

```bash
export AZURE_OPENAI_API_KEY=<your-api-key>
export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
pip install "instructor[openai]"
```

Azure OpenAI uses the standard `openai` SDK with an `AzureOpenAI` client — no
additional SDK is needed.

### See Also

- [Getting Started](../getting-started.md) - Quick start guide
- [from_provider Guide](../concepts/from_provider.md) - Detailed client configuration
- [Provider Examples](../index.md#provider-examples) - Quick examples for all providers

## Basic Usage

### One-line setup with `from_provider`

```python
import os
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("azure_openai/gpt-4o-mini")

user = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_model=User,
)

print(user)
# > User(name='Jason', age=25)
```

### Explicit client setup with `from_azure`

```python
import os
from openai import AzureOpenAI
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


raw_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
client = instructor.from_azure(raw_client)

user = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_model=User,
)

print(user)
# > User(name='Jason', age=25)
```

### Async example

```python
import asyncio
import os
from openai import AsyncAzureOpenAI
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


async def main():
    raw_client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    client = instructor.from_azure(raw_client)

    user = await client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[{"role": "user", "content": "Extract: Bob is 42 years old"}],
        response_model=User,
    )
    print(user)
    # > User(name='Bob', age=42)


asyncio.run(main())
```

## Supported Modes

Azure OpenAI supports the full OpenAI mode set:

| Mode | Description |
|------|-------------|
| `TOOLS` (default) | Uses the model's tool-calling feature for structured output |
| `JSON` | Uses `response_format={"type": "json_object"}` |
| `JSON_SCHEMA` | Uses OpenAI structured outputs (`response_format` with `json_schema`) |
| `MD_JSON` | Requests JSON via system prompt, extracts from markdown code blocks |
| `PARALLEL_TOOLS` | Parallel tool calling for `Iterable[Union[...]]` response models |

```python
from openai import AzureOpenAI
import instructor
from instructor import Mode

raw_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)

# Tool calling mode (default)
tools_client = instructor.from_azure(raw_client, mode=Mode.TOOLS)

# JSON mode
json_client = instructor.from_azure(raw_client, mode=Mode.JSON)

# Structured outputs (JSON_SCHEMA)
schema_client = instructor.from_azure(raw_client, mode=Mode.JSON_SCHEMA)
```

## Azure AI Foundry

Azure AI Foundry hosts both OpenAI models and non-OpenAI models (Mistral,
Llama, Phi) behind a single endpoint. Foundry-hosted models that expose an
OpenAI-compatible chat completions API work with `from_azure` out of the box:

```python
from openai import AzureOpenAI
import instructor

# Foundry endpoint for a non-OpenAI model
raw_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint="https://<your-foundry>.openai.azure.com",
)
client = instructor.from_azure(raw_client, mode=Mode.TOOLS)
```

## Authentication with Entra ID

For token-based authentication (Entra ID / Azure AD), create the
`AzureOpenAI` client with `azure-identity` and pass it to `from_azure`:

```bash
pip install azure-identity
```

```python
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
import instructor

credential = DefaultAzureCredential()
raw_client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_ad_token_provider=credential.get_token,
)
client = instructor.from_azure(raw_client)
```

## Nested Objects

```python
import os
import instructor
from pydantic import BaseModel


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


client = instructor.from_provider("azure_openai/gpt-4o-mini")

user = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    messages=[
        {
            "role": "user",
            "content": (
                "Extract: Jason is 25 years old. "
                "He lives at 123 Main St, New York, USA "
                "and has a summer house at 456 Beach Rd, Miami, USA."
            ),
        }
    ],
    response_model=User,
)

print(user)
#> User(
#>     name='Jason',
#>     age=25,
#>     addresses=[
#>         Address(street='123 Main St', city='New York', country='USA'),
#>         Address(street='456 Beach Rd', city='Miami', country='USA'),
#>     ]
#> )
```

## Additional Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [Azure AI Foundry Documentation](https://learn.microsoft.com/azure/ai-foundry/)
- [openai Python SDK](https://github.com/openai/openai-python)
