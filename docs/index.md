---
title: "Instructor - Get Structured LLM Outputs"
description: "Turn any LLM into a structured data extraction tool. Works with OpenAI, Anthropic, Google, Ollama, and 15+ providers."
---

# Turn any LLM into a structured data extractor

Instructor makes it dead simple to get structured, validated outputs from LLMs. Built on Pydantic, it gives you type-safe responses with automatic retries - all with a single line of code.

```python
import instructor
from pydantic import BaseModel

# One API for 15+ providers
client = instructor.from_provider("openai/gpt-4o-mini")

class User(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
)

print(user)  # User(name='Jason', age=25)
```

## Start in 30 seconds

=== "uv (recommended)"
    ```bash
    uv add instructor
    ```

=== "pip"
    ```bash
    pip install instructor
    ```

Then use the same API with any provider:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet-20241022")

# Google  
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")

# All work exactly the same!
```

## Why 3M+ developers choose Instructor

<div class="grid cards" markdown>

- :zap: **Dead simple API**
    
    Just add `response_model` to get structured outputs. No new syntax to learn.

- :shield: **Bulletproof validation**
    
    Powered by Pydantic. Automatic retries when validation fails.

- :arrows_counterclockwise: **Works everywhere**
    
    Same code for OpenAI, Anthropic, Google, Ollama, and 15+ providers.

- :rocket: **Production ready**
    
    Type hints, async support, streaming, and battle-tested by thousands.

</div>

## See it in action

### Extract complex data

```python
from typing import List
from pydantic import BaseModel
import instructor

client = instructor.from_provider("openai/gpt-4o-mini")

class Product(BaseModel):
    name: str
    features: List[str]
    price: float

class Receipt(BaseModel):
    store: str
    products: List[Product]
    total: float

receipt = client.chat.completions.create(
    response_model=Receipt,
    messages=[{
        "role": "user", 
        "content": "Extract: Bought 2 items at Acme Store. iPhone 15 Pro with 5G and ProMotion display for $999. AirPods Pro with noise cancellation for $249. Total: $1248."
    }]
)

print(receipt)
# Receipt(
#     store='Acme Store',
#     products=[
#         Product(name='iPhone 15 Pro', features=['5G', 'ProMotion display'], price=999.0),
#         Product(name='AirPods Pro', features=['noise cancellation'], price=249.0)
#     ],
#     total=1248.0
# )
```

### Stream partial results

```python
from instructor import Partial

# Stream objects as they're generated
for partial_user in client.chat.completions.create(
    response_model=Partial[User],
    stream=True,
    messages=[{"role": "user", "content": "Jason is 25 years old"}],
):
    print(partial_user)
    # User(name=None, age=None)
    # User(name='Jason', age=None) 
    # User(name='Jason', age=25)
```

### Automatic validation & retries

```python
from pydantic import field_validator

class Email(BaseModel):
    address: str
    
    @field_validator('address')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

# Instructor automatically retries on validation errors
email = client.chat.completions.create(
    response_model=Email,
    messages=[{"role": "user", "content": "Contact me at jason at example dot com"}],
    max_retries=3
)
print(email)  # Email(address='jason@example.com')
```

## Quick links

<div class="grid cards" markdown>

- :material-rocket-launch: **Getting Started**
    
    Learn the basics in 5 minutes
    
    [:octicons-arrow-right-16: Quick start](./concepts/models.md)

- :material-lightbulb: **Why Instructor?**
    
    See why developers love it
    
    [:octicons-arrow-right-16: Learn more](./why.md)

- :material-book-open-variant: **Examples**
    
    Production-ready code samples
    
    [:octicons-arrow-right-16: Browse cookbook](./examples/index.md)

- :material-chat: **Get Help**
    
    Join our community
    
    [:octicons-arrow-right-16: Discord](https://discord.gg/bD9YE9JArw)

</div>

## Trusted by industry leaders

!!! quote "What developers are saying"
    
    "Instructor eliminated 80% of our LLM integration code. What took days now takes hours."
    — **Senior Engineer, Fortune 500**
    
    "The best investment we made. Clean API, great docs, works flawlessly."
    — **CTO, YC Startup**
    
    "Finally, LLM outputs we can trust in production. Pydantic validation is a game-changer."
    — **ML Engineer, FAANG**

## Advanced features

### Multi-Provider Support
Use any LLM with the same clean API. Switch providers with one line.

[Learn more →](./integrations/index.md){ .md-button }

### Streaming & Partial Outputs
Process data as it's generated. Perfect for real-time applications.

[Learn more →](./concepts/partial.md){ .md-button }

### Async Support
Built for modern Python. Full asyncio support across all providers.

[Learn more →](./concepts/parallel.md){ .md-button }

### Smart Retries
Automatic retries with exponential backoff. Never lose data to transient errors.

[Learn more →](./concepts/retrying.md){ .md-button }

## Ready to build?

[Get Started →](./concepts/models.md){ .md-button .md-button--primary }
[Browse Examples →](./examples/index.md){ .md-button }

---

### Need help?

Run `instructor docs` to open documentation in your browser, or:

- 💬 [Join our Discord](https://discord.gg/bD9YE9JArw)
- 🐦 [Follow on Twitter](https://twitter.com/jxnlco)
- ⭐ [Star on GitHub](https://github.com/instructor-ai/instructor)