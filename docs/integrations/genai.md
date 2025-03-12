---
draft: False
date: 2025-03-11
title: "Structured outputs with Google's genai library, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Google's Generative AI (Gemini). Learn how to generate structured, type-safe outputs with Gemini models."
slug: genai
tags:
  - patching
authors:
  - instructor
---

# Structured outputs with Google Generative AI (Gemini), a complete guide w/ instructor

!!! warning "Recommended Library"

    We recommend using the `genai` library where possible to use the Gemini models. This is because it provides a unified interface for both the VertexAI and Gemini Developer APIs. It is also actively maintained and will be the default recomendation moving forward

This guide demonstrates how to use Google's Generative AI (Gemini) with Instructor to generate structured outputs. You'll learn how to use function calling with Gemini models to create type-safe responses.

Gemini is Google's family of multimodal large language models, with Gemini 1.5 supporting long context windows and function calling capabilities. Gemini's function calling makes it possible to obtain structured outputs using JSON schema.

## Quick Start

To use Instructor with Google's Generative AI, you'll need to install the required packages:

```bash
pip install "instructor[google-genai]"
```

## Gemini API Setup Guide

### Option 1: Google AI Studio / Gemini API

Set up your API key for direct access to Gemini models through Google AI Studio:

#### Method 1: Environment Variable

```bash
export GOOGLE_API_KEY='your-api-key-here'
```

#### Method 2: Direct Client Configuration

```python
import google.generativeai as genai
import instructor
from pydantic import BaseModel

# Initialize the client with your API key
client = genai.Client(api_key='YOUR_GEMINI_API_KEY')
```

### Option 2: Vertex AI (for Google Cloud customers)

Use Vertex AI to access Gemini models with enterprise features and integrations:

#### Step 1: Set Environment Variables

```bash
# Configure Vertex AI access
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'  # Or your preferred region
```

#### Step 2: Create Vertex AI Client

```python
import google.generativeai as genai
import instructor
from pydantic import BaseModel

# Initialize the client with Vertex AI configuration
client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1'  # Must match your GOOGLE_CLOUD_LOCATION
)
```

### Additional Notes

- For Vertex AI, ensure you have the necessary IAM permissions set up for your project
- API keys for Google AI Studio can be created at [https://aistudio.google.com/]
- Vertex AI and the Gemini Developer APIs require an active Google Cloud project with billing enabled

## Simple User Example (Sync)

```python
import os
from google import genai
import instructor
from pydantic import BaseModel

client = genai.Client()

# Enable instructor patches for Google Generative AI client
client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)

class User(BaseModel):
    name: str
    age: int

class Users(BaseModel):
    users: list[User]

# Create structured output
response = client.chat.completions.create(
    model="gemini-1.5-flash",
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=Users,
)

print(response)
#> Users(users=[User(name='Jason', age=25)])
```

## Simple User Example (Async)

```python
import os
from google import genai
import instructor
from pydantic import BaseModel
import asyncio

client = genai.Client()

# Enable instructor patches for async Google Generative AI client
client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS, use_async=True)


class User(BaseModel):
    name: str
    age: int


class Users(BaseModel):
    users: list[User]


async def extract_user():
    response = await client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=Users,
    )
    return response


# Run async function
users = asyncio.run(extract_user())
print(users)
# > Users(users=[User(name='Jason', age=25)])

```

## Formatting Messages

While we recommend using the OpenAI Chat Completions API message format, we support a broad range of potential message types at the moment as seen below.

**Note** : If you'd like to prefill or add some assistant messages, you must use the `model` role as per Gemini's API specification.

### Single String

Instructor supports various message formats with Gemini:

```python
import os
from google import genai
import instructor
from pydantic import BaseModel

client = genai.Client()

# Enable instructor patches for Google Generative AI client
client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)

class User(BaseModel):
    name: str
    age: int

class Users(BaseModel):
    users: list[User]

# Using a simple string as the message
response = client.chat.completions.create(
    model="gemini-1.5-flash",
    messages=["Ivan is 28 years old"],
    response_model=Users,
)

print(response)
#> Users(users=[User(name='Ivan', age=28)])
```

## Using System Messages

We've chosen to deal with system messages by concatenating all of the system messages that you provide. This is because Gemini only supports a single system message and so this is a deliberate choice.

We offer two main ways to format system messages

1. Using a kwarg of `system`
2. Using a message with a role of `system`

These are mutually exclusive and if you provide a kwarg of `system`, that will take precedence over the message with a role of `system`

### Using A Message

You can use one or more messages with a role of `system`

```python
import os
from google import genai
import instructor
from pydantic import BaseModel

client = genai.Client()

# Enable instructor patches for Google Generative AI client
client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)

class User(BaseModel):
    name: str
    age: int

class Users(BaseModel):
    users: list[User]

response = client.chat.completions.create(
    model="gemini-1.5-flash",
    messages=[
        {
            "role": "system",
            "content": "Ivan is 28 years old",
        },
        {
            "role": "user",
            "content": "Extract all users",
        },
    ],
    response_model=Users,
)

print(response)
#> Users(users=[User(name='Ivan', age=28)])
```

### Using a Keyword Argument

You can also use a keyword argument

```python
import os
from google import genai
import instructor
from pydantic import BaseModel

client = genai.Client()

# Enable instructor patches for Google Generative AI client
client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)

class User(BaseModel):
    name: str
    age: int

class Users(BaseModel):
    users: list[User]

response = client.chat.completions.create(
    model="gemini-1.5-flash",
    system="Ivan is 28 years old",
    messages=[
        {
            "role": "user",
            "content": "Extract all users",
        },
    ],
    response_model=Users,
)

print(response)
#> Users(users=[User(name='Ivan', age=28)])
```

## Messages

When it comes to messages, we support the normal messages format, using `genai.types.Content` and also using a string as input. Strings will always be converted to user messages by default.

```python
from google import genai
import instructor
from pydantic import BaseModel


client = genai.Client()

# Enable instructor patches for Google Generative AI client
client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)


class User(BaseModel):
    name: str
    age: int


class Users(BaseModel):
    users: list[User]


# Combining all three message formats in a single call
response = client.chat.completions.create(
    model="gemini-1.5-flash",
    messages=[
        # Format 1: Simple string
        "Ivan is 28 years old",
        # Format 2: Standard chat completion message format
        {
            "role": "user",
            "content": "Tiffany is 20 years old",
        },
        # Format 3: Using genai's Content type
        genai.types.Content(
            role="user",
            parts=[genai.types.Part.from_text(text="Jason is 25 years old")],
        ),
    ],
    response_model=Users,
)

print(response)
#> users=[User(name='Ivan', age=28), User(name='Tiffany', age=20), User(name='Jason', age=25)]
```

## Instructor Modes

For Google Generative AI, Instructor currently supports:

- `instructor.Mode.GENAI_TOOLS`: This uses Google's function calling API to return structured outputs to the client.

## Related Resources

- [Google Generative AI Documentation](https://googleapis.github.io/python-genai/#provide-a-list-types-content)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with the latest Google Generative AI API versions and models. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.
