---
title: "Structured outputs with Manifest, a complete guide with instructor"
description: "Learn how to use Instructor with Manifest, a smart LLM router that scores each request and routes it to the cheapest capable model. Get type-safe, structured outputs with automatic cost optimization."
---

# Structured outputs with Manifest, a complete guide with instructor

[Manifest](https://manifest.build/) is a smart model router for AI agents. It sits between your agent and your LLM providers, scores each request across 23 dimensions, and routes it to the cheapest model that can handle it. This guide shows you how to use Instructor with Manifest for type-safe, validated responses with automatic cost optimization.

## Quick Start

Install Instructor with the OpenAI extra (Manifest uses an OpenAI-compatible API):

```bash
pip install "instructor[openai]"
```

Set your Manifest API key:

```bash
export MANIFEST_API_KEY=<your-api-key>
```

You can get an API key at [app.manifest.build](https://app.manifest.build).

## Simple User Example (Sync)

The `"auto"` model lets Manifest automatically select the best model for each request.

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "manifest/auto",
    async_client=False,
)

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Ivan is 28 years old",
        },
    ],
    response_model=User,
)

print(resp)
#> name='Ivan' age=28
```

## Simple User Example (Async)

```python
import instructor
from pydantic import BaseModel
import asyncio


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "manifest/auto",
    async_client=True,
)


async def extract_user():
    user = await client.create(
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
    )
    return user


user = asyncio.run(extract_user())
print(user)
#> name='Jason' age=25
```

## Nested Object Example

```python
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


client = instructor.from_provider(
    "manifest/auto",
    async_client=False,
)

user = client.create(
    messages=[
        {
            "role": "user",
            "content": """
            Extract: Jason is 25 years old.
            He lives at 123 Main St, New York, USA
            and has a summer house at 456 Beach Rd, Miami, USA
        """,
        },
    ],
    response_model=User,
)

print(user)
#> name='Jason' age=25 addresses=[Address(street='123 Main St', city='New York', country='USA'), Address(street='456 Beach Rd', city='Miami', country='USA')]
```

## Streaming

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "manifest/auto",
    async_client=False,
)

user = client.create_partial(
    messages=[
        {
            "role": "user",
            "content": "Extract: Jason is 25 years old",
        },
    ],
    response_model=User,
)

for chunk in user:
    print(chunk)
    #> name=None age=None
    #> name='Jason' age=None
    #> name='Jason' age=25
```

## Self-Hosted

If you're running Manifest locally via Docker, point to your instance:

```python
client = instructor.from_provider(
    "manifest/auto",
    base_url="http://localhost:3001/v1",
    async_client=False,
)
```

## Related Resources

- [Manifest documentation](https://manifest.build/docs)
- [Instructor concepts](../concepts/index.md)
