---
description: Learn why Instructor is the simplest, most reliable way to get structured outputs from LLMs in Python.
---

# Why use Instructor?

Get structured, validated outputs from any LLM in just a few lines of code. Instructor makes language models work the way you already think about data in Python.

## The Problem with Raw LLM APIs

Working directly with language model APIs means:

- Writing complex JSON schemas by hand
- Parsing unstructured text outputs  
- No validation or type safety
- Handling errors and retries manually
- Different APIs for each provider

## Instructor Makes it Simple

With Instructor, you define what you want using Pydantic models - the same way you already define data in Python:

```python
from pydantic import BaseModel
from instructor import from_openai
from openai import OpenAI

class User(BaseModel):
    name: str
    age: int
    email: str

client = from_openai(OpenAI())

user = client.chat.completions.create(
    model="gpt-4",
    response_model=User,
    messages=[{"role": "user", "content": "Jason is 25, email: jason@example.com"}]
)

print(user)
# User(name='Jason', age=25, email='jason@example.com')
```

That's it. No JSON schemas, no parsing, no error handling boilerplate.

## Key Benefits

### 🎯 Type Safety Built In

Your IDE knows exactly what fields exist and their types. Catch errors before runtime.

```python
# Your IDE autocompletes these:
print(user.name)     # str
print(user.age)      # int
print(user.email)    # str
```

### ✅ Automatic Validation

Pydantic validates all outputs. Add custom validators to enforce business logic:

```python
from pydantic import field_validator

class User(BaseModel):
    name: str
    age: int
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Age must be between 0 and 150')
        return v
```

### 🔄 Smart Retries

When validation fails, Instructor automatically retries with the error message, helping the model self-correct:

```python
client.chat.completions.create(
    model="gpt-4",
    response_model=User,
    max_retries=3,  # Automatically retry up to 3 times
    messages=[...]
)
```

### 🌊 Streaming Support

Stream complex objects as they're generated:

```python
from instructor import Partial

for partial_user in client.chat.completions.create(
    model="gpt-4", 
    response_model=Partial[User],
    stream=True,
    messages=[...]
):
    print(partial_user)
    # User(name='Jason', age=None, email=None)
    # User(name='Jason', age=25, email=None)  
    # User(name='Jason', age=25, email='jason@example.com')
```

### 🔌 Works with Any Provider

Same code works across OpenAI, Anthropic, Google, Mistral, and more:

```python
from instructor import from_anthropic, from_gemini

# Same interface, different providers
anthropic_client = from_anthropic(Anthropic())
gemini_client = from_gemini(genai.GenerativeModel())
```

## Real-World Impact

### Save Time

- **Write 80% less code** compared to raw API usage
- **No more JSON schema debugging** - Pydantic handles it
- **Instant IDE support** with full autocomplete

### Reduce Errors

- **Catch issues at development time** with type checking
- **Automatic validation** prevents bad data in production
- **Self-healing retries** fix common LLM mistakes

### Scale Confidently

- **Battle-tested** by thousands of developers
- **3M+ monthly downloads** with Pydantic's 100M+ monthly downloads
- **Used in production** at Fortune 500 companies

## Who Uses Instructor?

Developers building:

- **Data extraction pipelines** - Parse documents into structured data
- **Content generation systems** - Create validated, formatted content
- **AI assistants** - Handle complex user queries with structured responses
- **Analytics tools** - Extract insights with guaranteed schema compliance
- **Automation workflows** - Chain LLM calls with type-safe data flow

## Getting Started Takes 30 Seconds

```bash
pip install instructor
```

Then just add `response_model` to your existing OpenAI code. That's it.

[Get Started →](../index.md#quick-start){ .md-button .md-button--primary }
