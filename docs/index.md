---
title: "Instructor - Structured Outputs for LLMs"
description: "Get reliable JSON from any LLM. Built on Pydantic for validation, type safety, and IDE support."
---

# Stop wrestling with LLM outputs

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

You're trying to extract structured data from LLMs, but you're stuck writing JSON schemas, handling validation errors, and parsing malformed responses. There's a better way.

## The problem

```python
# Without Instructor: Complex, error-prone, and different for each provider
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Extract user data..."}],
    tools=[{
        "type": "function", 
        "function": {
            "name": "extract_user",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    }]
)

# Parse response manually
tool_call = response.choices[0].message.tool_calls[0]
user_data = json.loads(tool_call.function.arguments)

# Hope the data is valid...
```

## The solution

```python
# With Instructor: Simple, type-safe, works with any provider
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

client = instructor.from_provider("openai/gpt-4")
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract user data..."}],
)

# user is validated, typed, and ready to use
print(user)  # User(name='John', age=25)
```

## Start in 30 seconds

```bash
pip install instructor
```

Then use with any provider:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")

# All use the exact same API
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Why developers choose Instructor

### 1. It just works

No more debugging JSON schemas or handling malformed responses. Define what you want with Pydantic, and Instructor handles the rest.

```python
from typing import List
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]

# Complex nested structures? No problem
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25, lives at 123 Main St, NYC, USA"}],
)
```

### 2. Built-in validation and retries

When the LLM returns invalid data, Instructor automatically retries with the validation error:

```python
from pydantic import field_validator

class Email(BaseModel):
    address: str
    
    @field_validator('address')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

# Instructor retries until it gets valid data
email = client.chat.completions.create(
    response_model=Email,
    messages=[{"role": "user", "content": "Contact me at john at example dot com"}],
    max_retries=3,
)
print(email)  # Email(address='john@example.com')
```

### 3. Streaming support

Get results as they're generated:

```python
from instructor import Partial

for partial_user in client.chat.completions.create(
    response_model=Partial[User],
    messages=[{"role": "user", "content": "..."}],
    stream=True,
):
    print(partial_user)
    # User(name=None, age=None)
    # User(name="John", age=None)
    # User(name="John", age=25)
```

## Real-world examples

### Extract data from documents

```python
from typing import List
from pydantic import BaseModel

class Invoice(BaseModel):
    invoice_number: str
    date: str
    total: float
    line_items: List[dict]

invoice = client.chat.completions.create(
    response_model=Invoice,
    messages=[{
        "role": "user",
        "content": "Extract invoice data from this image: ...",
    }],
)
```

### Classify and route support tickets

```python
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class SupportTicket(BaseModel):
    category: str
    priority: Priority
    summary: str
    requires_escalation: bool

ticket = client.chat.completions.create(
    response_model=SupportTicket,
    messages=[{
        "role": "user",
        "content": "Customer email: My payment is failing repeatedly...",
    }],
)
```

### Generate structured content

```python
class BlogPost(BaseModel):
    title: str
    tags: List[str]
    content: str
    seo_description: str

post = client.chat.completions.create(
    response_model=BlogPost,
    messages=[{
        "role": "user",
        "content": "Write a blog post about the future of AI",
    }],
)
```

## Is Instructor right for you?

**Use Instructor if you:**
- Need structured data from LLMs
- Want type safety and validation
- Work with multiple LLM providers
- Value simple, maintainable code

**You might not need Instructor if you:**
- Only need raw text outputs
- Are building simple chatbots
- Don't care about data validation

## Trusted by developers worldwide

- **3M+ monthly downloads**
- **10K+ GitHub stars**
- **100K+ developers**
- **1000+ contributors**

Used in production by teams at OpenAI, Google, Microsoft, and hundreds of startups.

## Get started

<div class="grid cards" markdown>

- **Quick Start**
    
    Learn the basics in 5 minutes
    
    [Get Started →](./concepts/models.md)

- **Examples**
    
    Copy-paste solutions for common tasks
    
    [Browse Examples →](./examples/index.md)

- **API Reference**
    
    Complete API documentation
    
    [View Reference →](./api.md)

- **Get Help**
    
    Join our community
    
    [Discord →](https://discord.gg/bD9YE9JArw)

</div>

## Common questions

### How is this different from JSON mode?

JSON mode gives you valid JSON, but not validated data. Instructor ensures your data matches your schema, automatically retries on errors, and provides type safety.

### Does it work with my LLM provider?

Yes! Instructor works with OpenAI, Anthropic, Google, Mistral, Cohere, and any provider that follows the OpenAI API format (including local models via Ollama).

### What about performance?

Instructor adds minimal overhead (<5ms) to your LLM calls. The time saved from not debugging malformed outputs far exceeds any performance cost.

### Can I use it in production?

Absolutely. Instructor is used in production by thousands of companies. It's battle-tested, well-maintained, and has a large community.

## Installation options

=== "uv"
    ```bash
    uv add instructor
    ```

=== "pip"
    ```bash
    pip install instructor
    ```

=== "poetry"
    ```bash
    poetry add instructor
    ```

## Type inference and IDE support

Instructor provides excellent IDE support with proper type inference:

```python
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)

# Your IDE knows these types
user.name  # str
user.age   # int
```

## Available in other languages

Instructor's simple API is available across many languages:

- [TypeScript](https://js.useinstructor.com) - Full-featured JavaScript/TypeScript library
- [Ruby](https://ruby.useinstructor.com) - Ruby implementation
- [Go](https://go.useinstructor.com) - Go implementation  
- [Elixir](https://hex.pm/packages/instructor) - Elixir implementation

## Learn more

- [Concepts](./concepts/index.md) - Core concepts and mental models
- [Cookbook](./examples/index.md) - Copy-paste examples for common tasks
- [Blog](./blog/index.md) - Tutorials and best practices
- [Hub](./hub/index.md) - Pre-built extractors and validators

## Ready to build?

[Get Started →](./concepts/models.md){ .md-button .md-button--primary }
[Browse Examples →](./examples/index.md){ .md-button }

---

Need help? [Join our Discord](https://discord.gg/bD9YE9JArw) • [Follow on Twitter](https://twitter.com/jxnlco) • [Star on GitHub](https://github.com/instructor-ai/instructor)