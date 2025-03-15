---
draft: False
date: 2025-03-15
title: "Structured outputs with Google's genai SDK"
description: "Learn how to use Instructor with Google's Generative AI SDK to extract structured data from Gemini models."
slug: genai
tags:
  - patching
authors:
  - instructor
---

# Structured Outputs with Google's genai SDK

!!! info "Recommended SDK"
The `genai` SDK is Google's recommended Python client for working with Gemini models. It provides a unified interface for both the Gemini API and Vertex AI. For detailed setup instructions, including how to use it with Vertex AI, please refer to the [official Google AI documentation for the GenAI SDK](https://googleapis.github.io/python-genai/).

This guide demonstrates how to use Instructor with Google's `genai` SDK to extract structured data from Gemini models.

## Installation

```bash
pip install "instructor[google-genai]"
```

## Basic Usage

Getting started with Instructor and the genai SDK is straightforward. Just create a Pydantic model defining your output structure, patch the genai client, and make your request with a response_model parameter:

```python
from google import genai
import instructor
from pydantic import BaseModel

# Define your Pydantic model
class User(BaseModel):
    name: str
    age: int

# Initialize and patch the client
client = genai.Client()
client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)

# Extract structured data
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_model=User,
)

print(response)  # User(name='Jason', age=25)
```

## Message Formatting

Genai supports multiple message formats, and Instructor seamlessly works with all of them. This flexibility allows you to use whichever format is most convenient for your application:

```python
# Single string (converted to user message)
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages="Jason is 25 years old",
    response_model=User,
)

# Standard format
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        {"role": "user", "content": "Jason is 25 years old"}
    ],
    response_model=User,
)

# Using genai's Content type
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        genai.types.Content(
            role="user",
            parts=[genai.types.Part.from_text("Jason is 25 years old")]
        )
    ],
    response_model=User,
)
```

### System Messages

System messages help set context and instructions for the model. With Gemini models, you can provide system messages in two different ways:

```python
# As a parameter
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    system="You are a data extraction assistant",
    messages=[{"role": "user", "content": "Jason is 25 years old"}],
    response_model=User,
)

# Or as a message with role "system"
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        {"role": "system", "content": "You are a data extraction assistant"},
        {"role": "user", "content": "Jason is 25 years old"}
    ],
    response_model=User,
)
```

## Template Variables

Template variables make it easy to reuse prompts with different values. This is particularly useful for dynamic content or when testing different inputs:

```python
import instructor
from google import genai
from google.genai import types
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_genai(genai.Client())

response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        {"role": "user", "content": "{{name}} is {{age}} years old"},

    ],
    response_model=User,
    context={"name": "Jason", "age": 25},
)

# We also support template variables using the default sdk
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    response_model=User,
    messages=[
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="Extract {{name}} is {{age}} years old")],
        ),  # type: ignore
    ],
    context={"name": "Jason", "age": 25},
)
```

## Validation and Retries

Instructor can automatically retry requests when validation fails, ensuring you get properly formatted data. This is especially helpful when enforcing specific data requirements:

```python
from typing import Annotated
from pydantic import AfterValidator, BaseModel

def uppercase_validator(v: str) -> str:
    if v.islower():
        raise ValueError("Name must be ALL CAPS")
    return v

class UserDetail(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)]
    age: int

response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "Extract: jason is 25 years old"}],
    response_model=UserDetail,
    max_retries=3,
)

print(response)  # UserDetail(name='JASON', age=25)
```

## Multimodal Capabilities

Gemini models excel at processing different types of media. Instructor makes it easy to extract structured data from multimodal inputs.

### Image Processing

Extract structured information from images with the same ease as text:

```python
from pydantic import BaseModel
import instructor

class ImageDescription(BaseModel):
    objects: list[str]
    scene: str

response = client.chat.completions.create(
    model="gemini-1.5-pro",
    messages=[
        {
            "role": "user",
            "content": [
                "Describe this image",
                # Image can be provided as:
                "path/to/image.jpg",  # Local path
                instructor.Image.from_path("path/to/image.jpg"),  # Helper
                "https://example.com/image.jpg",  # URL
            ]
        }
    ],
    response_model=ImageDescription,
)
```

### Audio Processing

Process audio files and extract structured data from their content:

```python
from pydantic import BaseModel
import instructor

class AudioContent(BaseModel):
    transcript: str
    speakers: int

response = client.chat.completions.create(
    model="gemini-1.5-pro",
    messages=[
        {
            "role": "user",
            "content": [
                "Transcribe this audio",
                instructor.Audio.from_path("path/to/audio.wav"),
                # Or from URL:
                instructor.Audio.from_url("https://example.com/audio.wav")
            ]
        }
    ],
    response_model=AudioContent,
)
```

## Streaming Responses

Streaming allows you to process responses incrementally rather than waiting for the complete result. Instructor provides two powerful streaming approaches with Gemini models.

### Iterable Streaming (Multiple Objects)

Receive a stream of complete, validated objects as they're generated:

```python
from collections.abc import Iterable
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

stream = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "Generate three different people"}],
    response_model=Iterable[Person],
    stream=True,
)

for person in stream:
    print(f"Received: {person}")
```

### Partial Streaming (Field-by-Field)

Watch a single complex object being built field-by-field in real time:

```python
from instructor.dsl.partial import Partial
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    occupation: str
    skills: list[str]

stream = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "Create a profile for a developer"}],
    response_model=Partial[Person],
    stream=True,
)

for partial_person in stream:
    print(f"Current state: {partial_person}")
```

## Async Support

Instructor provides full async support for the genai SDK, allowing you to make non-blocking requests in async applications:

```python
import asyncio

async def extract_user():
    client = genai.Client()
    client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS, use_async=True)

    response = await client.chat.completions.create(
        model="gemini-2.0-flash-001",
        messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
        response_model=User,
    )
    return response

user = asyncio.run(extract_user())
```
