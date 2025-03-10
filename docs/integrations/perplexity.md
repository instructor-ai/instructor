---
title: Structured Outputs with Perplexity AI and Pydantic
description: Learn how to use Perplexity AI with Instructor for structured JSON outputs using Pydantic models. Create type-safe, validated responses from Perplexity's Sonar models with Python.
---

# Structured Outputs with Perplexity AI

This guide demonstrates how to use Perplexity AI with Instructor to generate structured outputs. You'll learn how to use Perplexity's Sonar models with Pydantic to create type-safe, validated responses.

## Prerequisites

You'll need to sign up for a Perplexity account and get an API key. You can do that [here](https://www.perplexity.ai/).

```bash
export PERPLEXITY_API_KEY=<your-api-key-here>
pip install "instructor[perplexity]"
```

## Perplexity AI

Perplexity AI provides access to powerful language models through their API. Instructor supports structured outputs with Perplexity's models using the OpenAI-compatible API.

### Sync Example

```python
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel

# Initialize with API key
client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

# Enable instructor patches for Perplexity client
client = instructor.from_perplexity(client)


class User(BaseModel):
    name: str
    age: int


# Create structured output
user = client.chat.completions.create(
    model="sonar-medium-online",
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
import os
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel
import asyncio

# Initialize with API key
client = AsyncOpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

# Enable instructor patches for Perplexity client
client = instructor.from_perplexity(client)


class User(BaseModel):
    name: str
    age: int


async def extract_user():
    user = await client.chat.completions.create(
        model="sonar-medium-online",
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)
# > User(name='Jason', age=25)
```

### Nested Objects

```python
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel

# Initialize with API key
client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

# Enable instructor patches for Perplexity client
client = instructor.from_perplexity(client)


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


# Create structured output with nested objects
user = client.chat.completions.create(
    model="sonar-medium-online",
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
#> User(
#>     name='Jason',
#>     age=25,
#>     addresses=[
#>         Address(street='123 Main St', city='New York', country='USA'),
#>         Address(street='456 Beach Rd', city='Miami', country='USA')
#>     ]
#> )
```

## Supported Modes

Perplexity AI currently supports the following mode with Instructor:

- `PERPLEXITY_JSON`: Direct JSON response generation

```python
import os
from openai import OpenAI
import instructor
from instructor import Mode
from pydantic import BaseModel

# Initialize with API key
client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

# Enable instructor patches for Perplexity client with explicit mode
client = instructor.from_perplexity(
    client, 
    mode=Mode.PERPLEXITY_JSON
)


class User(BaseModel):
    name: str
    age: int


# Create structured output
user = client.chat.completions.create(
    model="sonar-medium-online",
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)
# > User(name='Jason', age=25)
```

## Additional Resources

- [Perplexity API Documentation](https://docs.perplexity.ai/)
- [Perplexity API Reference](https://docs.perplexity.ai/reference/post_chat_completions) 