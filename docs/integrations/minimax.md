---
title: Structured Outputs with MiniMax and Pydantic
description: Learn how to use MiniMax with Instructor for structured JSON outputs using Pydantic models. Create type-safe, validated responses from MiniMax's chat models with Python.
---

# Structured Outputs with MiniMax

This guide shows how to use [MiniMax](https://www.minimaxi.com/) with Instructor to produce validated, structured outputs. MiniMax exposes an OpenAI-compatible chat completions endpoint, so Instructor wraps it with the same ergonomics as any other OpenAI-style provider.

## Prerequisites

Sign up for a MiniMax account and create an API key in the console, then install the dependencies:

```bash
export MINIMAX_API_KEY=<your-api-key-here>
pip install "instructor"
```

### See Also

- [Getting Started](../getting-started.md) - Quick start guide
- [from_provider Guide](../concepts/from_provider.md) - Detailed client configuration
- [Provider Examples](../index.md#provider-examples) - Quick examples for all providers

# MiniMax

Instructor supports both tool-calling and JSON modes against MiniMax:

- `MINIMAX_TOOLS` — function/tool calling (default)
- `MINIMAX_JSON` — direct JSON responses

### Sync Example

```python
import instructor
from pydantic import BaseModel


client = instructor.from_provider("minimax/MiniMax-Text-01")


class User(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)
# > User(name='Jason', age=25)
```

### Async Example

```python
import asyncio
import instructor
from pydantic import BaseModel


async_client = instructor.from_provider(
    "minimax/MiniMax-Text-01",
    async_client=True,
)


class User(BaseModel):
    name: str
    age: int


async def extract_user() -> User:
    return await async_client.chat.completions.create(
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
    )


user = asyncio.run(extract_user())
print(user)
# > User(name='Jason', age=25)
```

### Nested Objects

```python
import instructor
from pydantic import BaseModel


client = instructor.from_provider("minimax/MiniMax-Text-01")


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


user = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": (
                "Extract: Jason is 25 years old. "
                "He lives at 123 Main St, New York, USA "
                "and has a summer house at 456 Beach Rd, Miami, USA"
            ),
        },
    ],
    response_model=User,
)

print(user)
```

## Supported Modes

```python
import instructor
from instructor import Mode
from pydantic import BaseModel


# Tool calling (default)
client = instructor.from_provider(
    "minimax/MiniMax-Text-01",
    mode=Mode.MINIMAX_TOOLS,
)


# JSON mode
client = instructor.from_provider(
    "minimax/MiniMax-Text-01",
    mode=Mode.MINIMAX_JSON,
)


class User(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)
print(user)
```

## Using a pre-configured client

You can also pass your own ``openai`` client pointed at MiniMax's base URL:

```python
import openai
import instructor


client = instructor.from_minimax(
    openai.OpenAI(
        api_key="<your-api-key>",
        base_url="https://api.minimax.chat/v1",
    )
)
```

## Additional Resources

- [MiniMax API Documentation](https://www.minimaxi.com/en/document)
