# Claude Agent SDK

This guide shows you how to use instructor with the Claude Agent SDK for structured outputs with agentic capabilities.

## Installation

```bash
pip install instructor claude-agent-sdk
```

Set your API key:
```bash
export ANTHROPIC_API_KEY=your-api-key
```

## Quick Start

```python
from instructor import from_claude_agent_sdk
from pydantic import BaseModel
import anyio

class User(BaseModel):
    name: str
    age: int

async def main():
    client = from_claude_agent_sdk()

    user = await client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: John is 25 years old"}]
    )
    print(user.name)  # John
    print(user.age)   # 25

anyio.run(main)
```

## Features

### Guaranteed JSON Schema Compliance

The Claude Agent SDK provides built-in JSON schema validation through its `output_format` option. This integration automatically:

1. Converts your Pydantic model to a JSON schema
2. Passes the schema to the Claude Agent SDK
3. Validates the response with Pydantic
4. Returns a type-safe model instance

### Automatic Message Conversion

The integration accepts instructor's standard message format and automatically converts it to a prompt for the Claude Agent SDK:

```python
# Standard instructor message format
user = await client.create(
    response_model=User,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Extract user info from: John, age 25"}
    ]
)
```

### Retry Support

The integration includes retry support for validation errors:

```python
user = await client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract user info..."}],
    max_retries=3  # Retry up to 3 times on validation errors
)
```

## Examples

### Contact Extraction

```python
from instructor import from_claude_agent_sdk
from pydantic import BaseModel, Field
from typing import Optional
import anyio

class ContactInfo(BaseModel):
    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    company: Optional[str] = Field(default=None, description="Company name")

async def main():
    client = from_claude_agent_sdk()

    email_text = """
    Hi, I'm Sarah Johnson from TechCorp Inc.
    Reach me at sarah.johnson@techcorp.com or (555) 123-4567.
    """

    contact = await client.create(
        response_model=ContactInfo,
        messages=[
            {"role": "user", "content": f"Extract contact info:\n\n{email_text}"}
        ]
    )

    print(f"Name: {contact.name}")
    print(f"Email: {contact.email}")
    print(f"Phone: {contact.phone}")
    print(f"Company: {contact.company}")

anyio.run(main)
```

### Sentiment Analysis

```python
from instructor import from_claude_agent_sdk
from pydantic import BaseModel, Field
from typing import List
import anyio

class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0.0, le=1.0)
    key_phrases: List[str] = Field(description="Key phrases influencing sentiment")

async def main():
    client = from_claude_agent_sdk()

    review = "This product is amazing! Best purchase I've ever made."

    result = await client.create(
        response_model=SentimentAnalysis,
        messages=[
            {"role": "user", "content": f"Analyze sentiment:\n\n{review}"}
        ]
    )

    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Key Phrases: {', '.join(result.key_phrases)}")

anyio.run(main)
```

### Entity Extraction

```python
from instructor import from_claude_agent_sdk
from pydantic import BaseModel, Field
from typing import List
import anyio

class Entity(BaseModel):
    name: str
    type: str  # person, organization, location, date, etc.

class EntityExtraction(BaseModel):
    entities: List[Entity]
    summary: str

async def main():
    client = from_claude_agent_sdk()

    text = """
    Apple CEO Tim Cook announced a meeting with French President Macron
    in Paris on March 15, 2025.
    """

    result = await client.create(
        response_model=EntityExtraction,
        messages=[
            {"role": "user", "content": f"Extract entities:\n\n{text}"}
        ]
    )

    for entity in result.entities:
        print(f"- {entity.name} ({entity.type})")

anyio.run(main)
```

## Key Differences from Other Providers

1. **Async and Sync Support**: The integration supports both async (default) and sync modes via `use_async=False`.

2. **Agentic Capabilities**: The Claude Agent SDK can perform agentic tasks with tool use, making it suitable for complex multi-step workflows.

3. **Built-in Schema Validation**: The SDK validates JSON output against the schema before returning, providing an extra layer of reliability.

4. **No Streaming Support**: Unlike other providers, Claude Agent SDK does not support streaming responses, `Partial` models, or `Iterable` models.

## Configuration Options

You can pass additional options when creating the client:

```python
client = from_claude_agent_sdk(
    model="claude-sonnet-4-20250514",  # Optional: specify model
    max_tokens=1024,                    # Optional: token limit
    temperature=0.0,                    # Optional: temperature
)
```

## Sync vs Async

By default, `from_claude_agent_sdk()` returns an async client. You can get a sync client by passing `use_async=False`:

### Async (default)

```python
from instructor import from_claude_agent_sdk
from pydantic import BaseModel
import anyio

class User(BaseModel):
    name: str
    age: int

async def main():
    client = from_claude_agent_sdk()  # async by default

    user = await client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: John is 25 years old"}]
    )
    print(user.name, user.age)

anyio.run(main)
```

### Sync

```python
from instructor import from_claude_agent_sdk
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

client = from_claude_agent_sdk(use_async=False)  # sync mode

user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 25 years old"}]
)
print(user.name, user.age)
```

Note: The sync version uses `anyio.run()` internally to execute the async Claude Agent SDK.

## Limitations

- No streaming support (`stream=True`, `Partial[Model]`, or `Iterable[Model]`)
- Use list fields in your model for multiple results
