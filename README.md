# Instructor

[![PyPI - Version](https://img.shields.io/pypi/v/instructor.svg)](https://pypi.org/project/instructor)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/instructor.svg)](https://pypi.org/project/instructor)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/instructor.svg)](https://pypi.org/project/instructor)
[![CI](https://github.com/jxnl/instructor/actions/workflows/ci.yml/badge.svg)](https://github.com/jxnl/instructor/actions/workflows/ci.yml)
[![License - MIT](https://img.shields.io/badge/license-MIT-9400d3.svg)](https://spdx.org/licenses/)
[![Discord](https://img.shields.io/discord/1192334452110659664?label=discord)](https://discord.gg/CV8sPM5k5Y)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/jxnlco)](https://twitter.com/jxnlco)

---

**Documentation**: [https://instructor-ai.github.io/instructor/](https://instructor-ai.github.io/instructor/)

**Source Code**: [https://github.com/jxnl/instructor](https://github.com/jxnl/instructor)

---

Instructor is a library for structured outputs with LLMs. It uses Pydantic to define structured outputs and automatically handles retries, validation, and error handling.

## Features

- **Response Models**: Define structured outputs using Pydantic models
- **Retry Management**: Automatically retry when the model fails to generate valid outputs
- **Validation**: Validate outputs against your Pydantic models
- **Multiple LLM Providers**: Support for OpenAI, Anthropic, Google, and more
- **Hooks**: Log and monitor LLM interactions
- **Streaming**: Stream partial results as they are generated
- **Async Support**: Async clients for all providers

## Installation

```bash
pip install instructor
```

## Usage

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

# Apply the patch to the OpenAI client
client = instructor.from_openai(OpenAI())

class UserDetail(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ]
)

print(user)
# UserDetail(name='Jason', age=25)
```

## Initializing Clients

Instructor provides a consistent interface for initializing clients across different providers:

```python
# OpenAI
import instructor
from openai import OpenAI
client = instructor.from_openai(OpenAI())

# Anthropic
import instructor
import anthropic
client = instructor.from_anthropic(anthropic.Anthropic())

# Google
import instructor
import google.generativeai as genai
client = instructor.from_genai(genai.GenerativeModel("gemini-pro"))

# Mistral
import instructor
from mistralai.client import MistralClient
client = instructor.from_mistral(MistralClient())

# Cohere
import instructor
import cohere
client = instructor.from_cohere(cohere.Client())

# LiteLLM
import instructor
import litellm
client = instructor.from_litellm(litellm.completion)

# Groq
import instructor
import groq
client = instructor.from_groq(groq.Groq())

# Perplexity
import instructor
from openai import OpenAI
client = instructor.from_perplexity(
    OpenAI(api_key="...", base_url="https://api.perplexity.ai")
)

# Writer
import instructor
from writerai import Writer
client = instructor.from_writer(Writer())

# AWS Bedrock
import instructor
import boto3
client = instructor.from_bedrock(boto3.client("bedrock-runtime"))

# Cerebras
import instructor
from cerebras.cloud.sdk import Cerebras
client = instructor.from_cerebras(Cerebras())

# Fireworks
import instructor
from fireworks.client import Fireworks
client = instructor.from_fireworks(Fireworks())

# VertexAI
import instructor
import vertexai.generative_models as gm
client = instructor.from_vertexai(gm.GenerativeModel(model_name="gemini-pro"))
```

You can also use the `from_provider` function to initialize a client based on a provider string:

```python
import instructor

# Sync clients
client = instructor.from_provider("openai/gpt-4")
client = instructor.from_provider("anthropic/claude-3-sonnet")

# Async clients
async_client = instructor.from_provider("openai/gpt-4", async_client=True)
```

## Hooks

Instructor provides a hooks system for logging and monitoring LLM interactions:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

# Define a hook
def log_prompt_hook(prompt, response, model_info):
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Model: {model_info}")

# Apply the hook to the client
client = instructor.from_openai(OpenAI())
client.on(log_prompt_hook)

# Use the client as normal
class UserDetail(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ]
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license.

## Provider Package Migration Guide

Starting from version 0.5.0, Instructor is migrating provider implementations to a more modular package structure. This change improves maintainability and organization of the codebase.

### What's Changed

Provider-specific implementations are being moved from flat `client_*.py` files to a structured `providers/` package:

```
instructor/
├── providers/
│   ├── __init__.py     # Provider registry and common utilities
│   ├── base.py         # Provider interface/protocol
│   ├── openai.py       # OpenAI implementation
│   └── ...             # Other providers
```

### How to Update Your Code

For most users, no changes are required as backward compatibility is maintained. However, if you're directly importing provider-specific functions, you can update your imports:

```python
# Old import
from instructor import from_openai

# New import (recommended)
from instructor.providers.openai import from_openai
```

The old imports will continue to work but will show deprecation warnings.
