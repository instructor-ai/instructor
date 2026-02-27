---
title: "Structured outputs with Inception Labs Mercury, a complete guide with instructor"
description: "Learn how to use Instructor with Inception Labs Mercury2 for type-safe, structured outputs."
---

# Structured outputs with Inception Labs Mercury, a complete guide with instructor

Inception Labs builds diffusion language models (dLLMs) that generate text in parallel rather than token-by-token. Mercury2 is their flagship model, notable for high-throughput structured output generation.

This guide covers everything you need to know about using Mercury2 with Instructor for type-safe, validated responses.

## Quick Start

Instructor comes with support for the OpenAI Client out of the box, so you don't need to install anything extra.

```bash
pip install "instructor"
```

Set your Inception Labs API key:

```bash
export INCEPTION_API_KEY='your-api-key-here'
```

## Simple User Example (Sync)

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("inceptionlabs/mercury-2")


class User(BaseModel):
    name: str
    age: int


# Create structured output
user = client.create(
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
    max_tokens=1024,
)

print(user)
#> User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
import instructor
from pydantic import BaseModel
import asyncio

client = instructor.from_provider(
    "inceptionlabs/mercury-2",
    async_client=True,
)


class User(BaseModel):
    name: str
    age: int


async def extract_user():
    user = await client.create(
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
        max_tokens=1024,
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)
#> User(name='Jason', age=25)
```

## Nested Example

```python
import instructor
from pydantic import BaseModel


client = instructor.from_provider("inceptionlabs/mercury-2")


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


# Create structured output with nested objects
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
    max_tokens=1024,
)

print(user)
#> {
#>     'name': 'Jason',
#>     'age': 25,
#>     'addresses': [
#>         {
#>             'street': '123 Main St',
#>             'city': 'New York',
#>             'country': 'USA'
#>         },
#>         {
#>             'street': '456 Beach Rd',
#>             'city': 'Miami',
#>             'country': 'USA'
#>         }
#>     ]
#> }
```

## Instructor Modes

Mercury2 supports two modes:

1. `instructor.Mode.INCEPTION_JSON` - Uses `response_format` with `json_schema` for structured outputs (default, recommended)
2. `instructor.Mode.INCEPTION_TOOLS` - Uses OpenAI-compatible tool calling

We recommend `INCEPTION_JSON` because Mercury2 has native support for schema-constrained JSON generation.

```python
import instructor

# Default mode (INCEPTION_JSON)
client = instructor.from_provider("inceptionlabs/mercury-2")

# Explicit tool calling mode
client = instructor.from_provider(
    "inceptionlabs/mercury-2",
    mode=instructor.Mode.INCEPTION_TOOLS,
)
```

## Direct Client Usage

You can also use the `from_inception` factory directly with an OpenAI client:

```python
import openai
import instructor

oai = openai.OpenAI(
    api_key="your-api-key",
    base_url="https://api.inceptionlabs.ai/v1",
)
client = instructor.from_inception(oai)
```

## Related Resources

- [Inception Labs Documentation](https://docs.inceptionlabs.ai/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with the latest Inception Labs API. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.
