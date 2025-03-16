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

We currently have two modes for Gemini

- `Mode.GENAI_TOOLS` : This leverages function calling under the hood and returns a structured response
- `Mode.GENAI_STRUCTURED_OUTPUTS` : This provides Gemini with a JSON Schema that it will use to respond in a structured format with

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
from google import genai
import instructor
from pydantic import BaseModel
from google.genai import types

# Define your Pydantic model
class User(BaseModel):
    name: str
    age: int

# Initialize and patch the client
client = genai.Client()
client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)

# Single string (converted to user message)
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages="Jason is 25 years old",
    response_model=User,
)

print(response)
# > name='Jason' age=25

# Standard format
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        {"role": "user", "content": "Jason is 25 years old"}
    ],
    response_model=User,
)

print(response)
# > name='Jason' age=25

# Using genai's Content type
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        genai.types.Content(
            role="user",
            parts=[genai.types.Part.from_text(text="Jason is 25 years old")]
        )
    ],
    response_model=User,
)

print(response)
# > name='Jason' age=25
```

### System Messages

System messages help set context and instructions for the model. With Gemini models, you can provide system messages in two different ways:

```python
from google import genai
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = genai.Client()
client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)

# As a parameter
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    system="Jason is 25 years old",
    messages=[{"role": "user", "content": "You are a data extraction assistant"}],
    response_model=User,
)

print(response)
# > name='Jason' age=25

# Or as a message with role "system"
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        {"role": "system", "content": "Jason is 25 years old"},
        {"role": "user", "content": "You are a data extraction assistant"},
    ],
    response_model=User,
)

print(response)
# > name='Jason' age=25

```

## Template Variables

Template variables make it easy to reuse prompts with different values. This is particularly useful for dynamic content or when testing different inputs:

```python
from google import genai
import instructor
from pydantic import BaseModel
from google.genai import types


# Define your Pydantic model
class User(BaseModel):
    name: str
    age: int


# Initialize and patch the client
client = genai.Client()
client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)

# Single string (converted to user message)
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=["{{name}} is {{ age }} years old"],
    response_model=User,
    context={
        "name": "Jason",
        "age": 25,
    },
)

print(response)
# > name='Jason' age=25

# Standard format
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "{{ name }} is {{ age }} years old"}],
    response_model=User,
    context={
        "name": "Jason",
        "age": 25,
    },
)

print(response)
# > name='Jason' age=25

# Using genai's Content type
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        genai.types.Content(
            role="user",
            parts=[genai.types.Part.from_text(text="{{name}} is {{age}} years old")],
        )
    ],
    response_model=User,
    context={
        "name": "Jason",
        "age": 25,
    },
)

print(response)
# > name='Jason' age=25
```

## Validation and Retries

Instructor can automatically retry requests when validation fails, ensuring you get properly formatted data. This is especially helpful when enforcing specific data requirements:

```python
from typing import Annotated
from pydantic import AfterValidator, BaseModel
import instructor
from google import genai


def uppercase_validator(v: str) -> str:
    if v.islower():
        raise ValueError("Name must be ALL CAPS")
    return v


class UserDetail(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)]
    age: int


client = instructor.from_genai(genai.Client())

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
from google import genai


class ImageDescription(BaseModel):
    objects: list[str]
    scene: str


client = instructor.from_genai(genai.Client())

# Method 1 : Using a local file path
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        {
            "role": "user",
            "content": [
                "Describe this image",
                "./scones.jpg",  # Local path
            ],
        }
    ],
    response_model=ImageDescription,
)
print(response)
#> objects=['cookies', 'coffee', 'blueberries', 'flowers'] scene='food photography'

# Method 2 : Using instructor's image method to explicitly load an image
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        {
            "role": "user",
            "content": [
                "Describe this image",
                instructor.Image.from_path("path/to/image.jpg"),  # Helper
            ],
        }
    ],
    response_model=ImageDescription,
)
print(response)
#> objects=['cookies', 'coffee', 'blueberries', 'flowers'] scene='food photography'

# Method 3 : Providing a image url
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        {
            "role": "user",
            "content": [
                "Describe this image",
                "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg",  # URL
            ]
        }
    ],
    response_model=ImageDescription,
)
print(response)
#> objects=['blueberries'] scene='blueberry field'
```

### Audio Processing

Process audio files and extract structured data from their content:

```python
from pydantic import BaseModel
import instructor
from google import genai
from google.genai import types


class ImageDescription(BaseModel):
    objects: list[str]
    scene: str


client = instructor.from_genai(genai.Client())


class AudioContent(BaseModel):
    summary: str

# Method 1 : Reading from a path itself
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    system="You are a helpful assistant that can extract and summarise the content of an audio file according to the schema provided. You must return a function call with the schema provided.",
    messages=[
        {
            "role": "user",
            "content": [
                "Extract and summarise the content of this audio",
                instructor.Audio.from_path("./pixel.mp3"),
            ],
        }
    ],
    response_model=AudioContent,
)
print(response)
# > summary='The Made by Google podcast discusses the Pixel feature drops with Aisha Sharif and DeCarlos Love. They discuss the importance of devices improving over time and the inte...

# Method 2 : Using a url

response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    system="You are a helpful assistant that can extract and summarise the content of an audio file according to the schema provided. You must return a function call with the schema provided.",
    messages=[
        {
            "role": "user",
            "content": [
                "Extract and summarise the content of this audio",
                instructor.Audio.from_url(
                    "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"
                ),
            ],
        }
    ],
    response_model=AudioContent,
)
print(response)
#> summary="Abraham Lincoln's Gettysburg Address, beginning with 'Four score and seven years ago' and discussing the Civil War's test of a nation dedicated to equality"
```

## Using Files

Our API integration also supports the use of files

```python
from google import genai
import instructor
from pydantic import BaseModel


class Summary(BaseModel):
    summary: str


client = genai.Client()
client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)

file1 = client.files.upload(
    file="./gettysburg.wav",
)

# As a parameter
response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    system="Summarise the audio file.",
    messages=[
        file1,
    ],
    response_model=Summary,
)

print(response)
# > summary="Abraham Lincoln's Gettysburg Address commences by stating that 87 years prior, the founding fathers created a new nation based on liberty and equality. It goes on to say that the Civil War is testing whether a nation so conceived can survive."
```

## Streaming Responses

> **Note:** Streaming functionality is currently only available when using the `Mode.GENAI_STRUCTURED_OUTPUTS` mode with Gemini models. Other modes like `tools` do not support streaming at this time.

Streaming allows you to process responses incrementally rather than waiting for the complete result. This is extremely useful for making UI changes feel instant and responsive.

### Partial Streaming

Receive a stream of complete, validated objects as they're generated:

```python
from pydantic import BaseModel
import instructor
from google import genai


client = instructor.from_genai(
    genai.Client(), mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
)


class Person(BaseModel):
    name: str
    age: int


class PersonList(BaseModel):
    people: list[Person]


stream = client.chat.completions.create_partial(
    model="gemini-2.0-flash-001",
    system="You are a helpful assistant. You must return a function call with the schema provided.",
    messages=[
        {
            "role": "user",
            "content": "Ivan is 20 years old, Jason is 25 years old, and John is 30 years old",
        }
    ],
    response_model=PersonList,
)

for extraction in stream:
    print(extraction)
    # > people=[PartialPerson(name='Ivan', age=None)]
    # > people=[PartialPerson(name='Ivan', age=20), PartialPerson(name='Jason', age=25), PartialPerson(name='John', age=None)]
    # > people=[PartialPerson(name='Ivan', age=20), PartialPerson(name='Jason', age=25), PartialPerson(name='John', age=30)]

```

## Async Support

Instructor provides full async support for the genai SDK, allowing you to make non-blocking requests in async applications:

```python
import asyncio

import instructor
from google import genai
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


async def extract_user():
    client = genai.Client()
    client = instructor.from_genai(
        client, mode=instructor.Mode.GENAI_TOOLS, use_async=True
    )

    response = await client.chat.completions.create(
        model="gemini-2.0-flash-001",
        messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
        response_model=User,
    )
    return response


print(asyncio.run(extract_user()))
#> name = Jason age= 25
```
