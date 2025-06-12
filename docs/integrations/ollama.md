---
draft: False
date: 2024-02-08
title: "Structured outputs with Ollama, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Ollama. Learn how to generate structured, type-safe outputs with Ollama."
slug: ollama
tags:
  - patching
  - open source
authors:
  - jxnl
---

# Structured outputs with Ollama, a complete guide w/ instructor

This guide demonstrates how to use Ollama with Instructor to generate structured outputs. You'll learn how to use JSON schema mode with local LLMs to create type-safe responses.

Open-source LLMS are gaining popularity, and the release of Ollama's OpenAI compatibility later it has made it possible to obtain structured outputs using JSON schema.

By the end of this blog post, you will learn how to effectively utilize instructor with ollama. But before we proceed, let's first explore the concept of patching.

<!-- more -->

## Patching

Instructor's patch enhances a openai api it with the following features:

- `response_model` in `create` calls that returns a pydantic model
- `max_retries` in `create` calls that retries the call if it fails by using a backoff strategy

!!! note "Learn More"

    To learn more, please refer to the [docs](../index.md). To understand the benefits of using Pydantic with Instructor, visit the tips and tricks section of the [why use Pydantic](../why.md) page.

## Getting Started

Start by downloading [Ollama](https://ollama.ai/download), and then pull a model such as Llama 2 or Mistral.

!!! tip "Make sure you update your `ollama` to the latest version!"

```
ollama pull llama2
```

## Quick Setup with Auto Client

The easiest way to use Ollama with Instructor is through the auto client:

```python
import instructor
from pydantic import BaseModel, Field
from typing import List

class Character(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")

# Simple setup - automatically configured for Ollama
client = instructor.from_provider("ollama/llama2")

resp = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Tell me about Harry Potter",
        }
    ],
    response_model=Character,
)

print(resp.model_dump_json(indent=2))
```

You can also customize the base URL if your Ollama server is running on a different host or port:

```python
client = instructor.from_provider(
    "ollama/llama2", 
    base_url="http://your-host:11434/v1"
)
```

### Async Support

The auto client also supports async operations:

```python
import instructor
import asyncio
from pydantic import BaseModel, Field
from typing import List

class Character(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")

# Create async client
async_client = instructor.from_provider("ollama/llama2", async_client=True)

async def main():
    resp = await async_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Tell me about Harry Potter",
            }
        ],
        response_model=Character,
    )
    print(resp.model_dump_json(indent=2))

# Run the async function
asyncio.run(main())
```

## Manual Setup

If you need more control over the client configuration, you can manually set up the client:

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

import instructor


class Character(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")


# enables `response_model` in create call
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

resp = client.chat.completions.create(
    model="llama2",
    messages=[
        {
            "role": "user",
            "content": "Tell me about the Harry Potter",
        }
    ],
    response_model=Character,
)
print(resp.model_dump_json(indent=2))
"""
{
  "name": "Harry James Potter",
  "age": 37,
  "fact": [
    "He is the chosen one.",
    "He has a lightning-shaped scar on his forehead.",
    "He is the son of James and Lily Potter.",
    "He attended Hogwarts School of Witchcraft and Wizardry.",
    "He is a skilled wizard and sorcerer.",
    "He fought against Lord Voldemort and his followers.",
    "He has a pet owl named Snowy."
  ]
}
"""
```
