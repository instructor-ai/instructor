---
title: Unified Provider Interface in Instructor
description: Learn how to use Instructor's new provider initialization to work with any LLM provider using a simple, consistent interface.
date: 2024-03-20
categories:
  - Features
  - Providers
tags:
  - initialization
  - providers
  - tutorial
---

# Unified Provider Interface in Instructor

Instructor now offers a simpler way to work with different LLM providers through a unified initialization interface. This feature makes it easy to switch between providers while keeping your code clean and consistent.

## Basic Usage

The new `from_provider` function lets you create an Instructor client for any supported provider using a simple string:

```python
import instructor
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# Create clients for different providers
openai_client = instructor.from_provider("openai/gpt-4")
anthropic_client = instructor.from_provider("anthropic/claude-3-sonnet")
google_client = instructor.from_provider("google/gemini-pro")

# Use the same interface for all providers
user_info = openai_client.chat.completions.create(
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John Doe is 30 years old"}]
)
```

## Async Support

Need to work with async code? Just set `async_client=True`:

```python
async_client = instructor.from_provider("openai/gpt-4", async_client=True)

async def extract_info():
    return await async_client.chat.completions.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": "John Doe is 30 years old"}]
    )
```

## Supported Providers

You can use any of these provider strings:

```python
# Major cloud providers
instructor.from_provider("openai/gpt-4")              # OpenAI
instructor.from_provider("anthropic/claude-3-sonnet")  # Anthropic
instructor.from_provider("google/gemini-pro")         # Google
instructor.from_provider("vertexai/gemini-pro")       # Vertex AI
instructor.from_provider("bedrock/anthropic.claude")  # AWS Bedrock
instructor.from_provider("genai/gemini-pro")         # Google GenAI

# Additional providers
instructor.from_provider("mistral/mistral-large")    # Mistral
instructor.from_provider("cohere/command")           # Cohere
instructor.from_provider("perplexity/pplx-7b")       # Perplexity
instructor.from_provider("groq/llama2-70b")          # Groq
instructor.from_provider("writer/writer")            # Writer
instructor.from_provider("cerebras/cerebras-gpt")    # Cerebras
instructor.from_provider("fireworks/llama-v2-70b")   # Fireworks
```

## Required Dependencies

Each provider requires its own Python package. Install only what you need:

```bash
# For OpenAI
pip install "instructor[openai]"

# For Anthropic
pip install "instructor[anthropic]"

# For multiple providers
pip install "instructor[openai,anthropic,google]"
```

## Benefits

1. **Consistent Interface**: Use the same code pattern across all providers
2. **Easy Switching**: Change providers by changing a single string
3. **Clean Code**: No need to import provider-specific packages in your main code
4. **Type Safety**: Full type hints and IDE support
5. **Async Support**: Built-in async capabilities for all providers

## Error Handling

The `from_provider` function includes helpful error messages:

```python
try:
    client = instructor.from_provider("unknown/model")
except ValueError as e:
    print(e)  # "Unsupported provider: unknown"

try:
    client = instructor.from_provider("openai-gpt4")  # Missing slash
except ValueError as e:
    print(e)  # "Model string must be in format 'provider/model-name'"
```

## Migration Guide

If you're using provider-specific initialization, here's how to switch:

```python
# Old way
from openai import OpenAI
client = instructor.from_openai(OpenAI())

# New way
client = instructor.from_provider("openai/gpt-4")
```

The new approach is shorter, cleaner, and more consistent across providers.

## Next Steps

- Check out our [provider-specific documentation](../integrations/index.md) for detailed features
- Join our [Discord community](https://discord.gg/bD9YE9JArw) for help and updates
- Star us on [GitHub](https://github.com/instructor-ai/instructor) to stay updated

The unified provider interface makes it easier than ever to work with multiple LLM providers in your projects. Try it out and let us know what you think! 