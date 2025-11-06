---
title: Getting Started
description: A step-by-step guide to getting started with Instructor for structured outputs from LLMs
---

# Getting Started with Instructor

This guide will walk you through the basics of using Instructor to extract structured data from language models. By the end, you'll understand how to:

1. Install and set up Instructor
2. Extract basic structured data
3. Handle validation and errors
4. Work with streaming responses
5. Use different LLM providers

## Installation

First, install Instructor with Google GenAI support:

```bash
uv add "instructor[google-genai]"
```

Add extras only when you need another provider:

```bash
uv add "instructor[anthropic]"
uv add "instructor[mistralai]"
uv add "instructor[litellm]"
```

## Setting Up Environment

Set your API keys as environment variables:

```bash
# For Google GenAI (recommended)
export GOOGLE_API_KEY=your_google_key

# For Vertex AI deployments
export GOOGLE_CLOUD_PROJECT=your_project_id
export GOOGLE_CLOUD_LOCATION=your_region

# For other providers, set their API keys when you need them
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Your First Structured Output

Let's start with a simple example using Google GenAI and the async client:

```python
import asyncio
import instructor
from instructor import Mode
from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str
    age: int


async def main() -> None:
    client = instructor.from_provider(
        "google/gemini-2.5-flash",
        async_client=True,
        mode=Mode.GENAI_STRUCTURED_OUTPUTS,
    )

    user_info = await client.chat.completions.create(
        response_model=UserInfo,
        messages=[
            {"role": "user", "content": "John Doe is 30 years old."}
        ],
    )

    print(f"Name: {user_info.name}, Age: {user_info.age}")


asyncio.run(main())
```

This example demonstrates the core workflow:
1. Define a Pydantic model for your output structure
2. Patch your LLM client with Instructor
3. Request structured output using the `response_model` parameter

## Validation and Error Handling

Instructor leverages Pydantic's validation to ensure your data meets requirements:

```python
import asyncio
import instructor
from instructor import Mode
from pydantic import BaseModel, Field, field_validator


class User(BaseModel):
    name: str
    age: int = Field(gt=0, lt=120)

    @field_validator("name")
    def name_must_have_space(cls, value: str) -> str:
        if " " not in value:
            raise ValueError("Name must include first and last name")
        return value


async def main() -> None:
    client = instructor.from_provider(
        "google/gemini-2.5-flash",
        async_client=True,
        mode=Mode.GENAI_STRUCTURED_OUTPUTS,
    )

    user = await client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Tom is 25 years old."}],
        max_retries=3,
    )

    print(user)


asyncio.run(main())
```

## Working with Complex Models

Instructor works seamlessly with nested Pydantic models:

```python
import asyncio
import instructor
from typing import List
from instructor import Mode
from pydantic import BaseModel


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str


class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]


async def main() -> None:
    client = instructor.from_provider(
        "google/gemini-2.5-flash",
        async_client=True,
        mode=Mode.GENAI_STRUCTURED_OUTPUTS,
    )

    person = await client.chat.completions.create(
        response_model=Person,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract: John Smith is 35 years old. "
                    "He has homes at 123 Main St, Springfield, IL 62704 and "
                    "456 Oak Ave, Chicago, IL 60601."
                ),
            }
        ],
    )

    print(person)


asyncio.run(main())
```

## Streaming Responses

For larger responses or better user experience, use streaming:

```python
import asyncio
import instructor
from instructor import Mode
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    summary: str


async def main() -> None:
    client = instructor.from_provider(
        "google/gemini-2.5-flash",
        async_client=True,
        mode=Mode.GENAI_STRUCTURED_OUTPUTS,
    )

    stream = client.chat.completions.create_partial(
        response_model=Person,
        messages=[
            {
                "role": "user",
                "content": "Extract a detailed person profile for John Smith, 35, who lives in Chicago and Springfield.",
            }
        ],
        stream=True,
    )

    async for partial in stream:
        print(partial)


asyncio.run(main())
```

## Using Different Providers

Instructor supports multiple LLM providers. Here's how to use Anthropic:

```python
import instructor
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# Create an instructor client with from_provider
client = instructor.from_provider("anthropic/claude-3-opus-20240229")

user_info = client.messages.create(
    max_tokens=1024,
    response_model=UserInfo,
    messages=[
        {"role": "user", "content": "John Doe is 30 years old."}
    ],
)

print(f"Name: {user_info.name}, Age: {user_info.age}")
```

## Next Steps

Now that you've mastered the basics, here are some next steps:

- Learn about [Mode settings](./concepts/patching.md) for different LLM providers
- Explore [advanced validation](./concepts/reask_validation.md) to ensure data quality
- Check out the [Cookbook examples](./examples/index.md) for real-world applications
- See how to [use hooks](./concepts/hooks.md) for monitoring and debugging

For more detailed information on any topic, visit the [Concepts](./concepts/index.md) section.

If you have questions or need help, join our [Discord community](https://discord.gg/bD9YE9JArw) or check the [GitHub repository](https://github.com/jxnl/instructor).