---
title: "Structured outputs with DeepSeek, a complete guide w/ instructor"
description: "Learn how to use Instructor with DeepSeek's models for type-safe, structured outputs."
---

# DeepSeek Integration with Instructor

DeepSeek is a Chinese company that provides AI models and services. They're most notable for the deepseek coder and chat model and most recently, the R1 reasoning model.

This guide covers everything you need to know about using DeepSeek with Instructor for type-safe, validated responses.

## Quick Start

Instructor comes with support for the OpenAI Client out of the box, so you don't need to install anything extra.

```bash
=== "UV (Recommended)"
    ```bash
Let's first install the instructor client with deepseek support:

=== "UV (Recommended)"
    ```bash
    # Install UV if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install instructor
    uv pip install "instructor[deepseek]"
    ```

=== "pip"
    ```bash
    pip install "instructor[deepseek]"
    ```






⚠️ **Important**: You must set your DeepSeek API key before using the client. You can do this in two ways:

1. Set the environment variable:

```bash
export DEEPSEEK_API_KEY='your-api-key-here'
```

2. Or provide it directly to the client:

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
```

## Simple User Example (Sync)

```python
import os
from openai import OpenAI
from pydantic import BaseModel
import instructor

client = instructor.from_openai(
    OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
)


class User(BaseModel):
    name: str
    age: int


# Create structured output
user = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)
# > name='Jason' age=25
```

## Simple User Example (Async)

```python
import os
import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel
import instructor

client = instructor.from_openai(
    AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )
)


class User(BaseModel):
    name: str
    age: int


async def extract_user():
    user = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)
# > name='Jason' age=25

```

## Nested Example

```python
from pydantic import BaseModel
import os
from openai import OpenAI
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


# Initialize with API key
client = instructor.from_openai(
    OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
)


# Create structured output with nested objects
user = client.chat.completions.create(
    model="deepseek-chat",
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

## Streaming Support

Instructor has two main ways that you can use to stream responses out

1. **Iterables**: These are useful when you'd like to stream a list of objects of the same type (Eg. use structured outputs to extract multiple users)
2. **Partial Streaming**: This is useful when you'd like to stream a single object and you'd like to immediately start processing the response as it comes in.

### Partials

```python
from pydantic import BaseModel
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel


# Initialize with API key
client = instructor.from_openai(
    OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
)


class User(BaseModel):
    name: str
    age: int
    bio: str


user = client.chat.completions.create_partial(
    model="deepseek-chat",
    messages=[
        {
            "role": "user",
            "content": "Create a user profile for Jason and a one sentence bio, age 25",
        },
    ],
    response_model=User,
)

for user_partial in user:
    print(user_partial)


# > name='Jason' age=None bio='None'
# > name='Jason' age=25 bio='A tech'
# > name='Jason' age=25 bio='A tech enthusiast'
# > name='Jason' age=25 bio='A tech enthusiast who loves coding, gaming, and exploring new'
# > name='Jason' age=25 bio='A tech enthusiast who loves coding, gaming, and exploring new technologies'

```

### Iterable Example

```python
from pydantic import BaseModel
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel


# Initialize with API key
client = instructor.from_openai(
    OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
)


class User(BaseModel):
    name: str
    age: int


# Extract multiple users from text
users = client.chat.completions.create_iterable(
    model="deepseek-chat",
    messages=[
        {
            "role": "user",
            "content": """
            Extract users:
            1. Jason is 25 years old
            2. Sarah is 30 years old
            3. Mike is 28 years old
        """,
        },
    ],
    response_model=User,
)

for user in users:
    print(user)

    #> name='Jason' age=25
    #> name='Sarah' age=30
    #> name='Mike' age=28
```

## Instructor Modes

We suggest using the `Mode.Tools` mode for Deepseek which is the default mode for the `from_openai` method.

## Related Resources

- [DeepSeek Documentation](https://api-docs.deepseek.com/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with the latest OpenAI API versions and models. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.
