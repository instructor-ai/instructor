---
title: Structured outputs with Azure OpenAI, a complete guide w/ instructor
description: Learn how to use Azure OpenAI with instructor for structured outputs, including async/sync implementations, streaming, and validation.
---

# Azure OpenAI Integration Guide

This guide demonstrates how to use Azure OpenAI with instructor for structured outputs. Azure OpenAI provides the same powerful models as OpenAI but with enterprise-grade security and compliance features through Microsoft Azure.

!!! tips "Key Features"
    - Enterprise-grade security and compliance
    - Regional data residency
    - Azure Active Directory integration
    - Custom deployment configurations
    - Usage monitoring and quotas

## Installation

First, install the required dependencies:

```bash
pip install "instructor[azure]"
```

## Authentication

To use Azure OpenAI, you'll need:

1. Azure OpenAI endpoint
2. API key
3. Deployment name

```python
import os
from openai import AzureOpenAI
import instructor

# Configure Azure OpenAI client
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-15-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
)

# Patch the client with instructor
client = instructor.from_openai(client)
```

## Basic Usage

Here's a simple example using a Pydantic model:

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Synchronous usage
user = client.chat.completions.create(
    model="gpt-4",  # Your deployment name
    messages=[
        {"role": "user", "content": "John is 30 years old"}
    ],
    response_model=User
)
print(f"Name: {user.name}, Age: {user.age}")
```

## Async Implementation

Azure OpenAI supports async operations:

```python
import asyncio
from openai import AsyncAzureOpenAI

async_client = AsyncAzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-15-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
)
async_client = instructor.from_openai(async_client)

async def get_user_async():
    user = await async_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "John is 30 years old"}
        ],
        response_model=User
    )
    return user

# Run async function
user = asyncio.run(get_user_async())
```

## Nested Models

Azure OpenAI handles complex nested structures:

```python
from typing import List

class Address(BaseModel):
    street: str
    city: str
    country: str

class UserWithAddress(BaseModel):
    name: str
    age: int
    addresses: List[Address]

user = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": """
        John is 30 years old and has two addresses:
        1. 123 Main St, New York, USA
        2. 456 High St, London, UK
        """}
    ],
    response_model=UserWithAddress
)
```

## Streaming Support

Azure OpenAI supports streaming responses:

```python
from instructor import from_openai
from pydantic import BaseModel, Field

class StreamingResponse(BaseModel):
    content: str = Field(description="Partial content being streamed")

async def stream_response():
    async for partial in async_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Write a long story about AI"}],
        response_model=StreamingResponse,
        stream=True
    ):
        print(partial.content, end="", flush=True)
```

## Iterable Responses

Process multiple items efficiently:

```python
from typing import Iterator

class Item(BaseModel):
    name: str
    category: str

def process_items() -> Iterator[Item]:
    return client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": """
            List of items:
            1. iPhone (Electronics)
            2. Book (Literature)
            3. Laptop (Electronics)
            """}
        ],
        response_model=Item,
        iterator=True
    )

for item in process_items():
    print(f"{item.name}: {item.category}")
```

## Instructor Hooks

Customize behavior with hooks:

```python
from instructor import from_openai

def log_completion(completion, model_response):
    print(f"Model used: {completion.model}")
    print(f"Tokens used: {completion.usage.total_tokens}")

client = from_openai(
    client,
    mode="tools",
    after_completion=log_completion
)
```

## Best Practices

1. **Environment Variables**: Store credentials securely
2. **Error Handling**: Implement proper error handling
3. **Deployment Names**: Use consistent deployment names
4. **API Versions**: Stay updated with latest versions
5. **Monitoring**: Use Azure Monitor for tracking

```python
try:
    result = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Process this"}],
        response_model=YourModel
    )
except Exception as e:
    print(f"Error: {e}")
    # Implement proper error handling
```

## Limitations

- Streaming support depends on the model deployment
- Some features may require specific API versions
- Function calling support varies by model

## Additional Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Instructor Documentation](https://instructor-ai.github.io/instructor/)
- [Azure OpenAI Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
