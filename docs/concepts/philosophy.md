---
title: Philosophy - Simple, Transparent, Flexible
description: Learn the core principles behind Instructor - simplicity over complexity, transparency over magic, and flexibility over constraints.
---

# Philosophy

Instructor is built on a simple idea: **Python developers should be able to get structured data from language models without learning new abstractions.**

## Core Principles

### 1. Simplicity First

We believe the best tools are invisible. You shouldn't need to learn a new framework to use LLMs effectively.

```python
# This is all you need to know:
response_model=YourPydanticModel
```

That's it. No DSLs, no complex configurations, no new paradigms. Just Pydantic models you already know.

### 2. Zero Lock-in

Your code should work with or without Instructor:

```python
# With Instructor
client = instructor.from_provider("openai/gpt-4")
user = client.chat.completions.create(
    response_model=User,  # Just remove this line to go back to raw API
    messages=[...]
)

# Without Instructor - use provider's native client
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...]
)
```

We patch, we don't wrap. Your escape hatch is always one line away.

### 3. Transparent by Design

We don't hide what we're doing. Instructor:

- Generates minimal, readable prompts
- Shows you exactly what's being sent to the LLM
- Lets you override any behavior you need
- Keeps the LLM's raw response accessible

```python
# See what Instructor sends
import instructor
instructor.logfire.configure()  # Full observability

# Access raw responses
completion = client.chat.completions.create(
    response_model=User,
    messages=[...], 
)
print(completion._raw_response)  # Original API response
```

### 4. Composition Over Configuration

Instead of configuration files and complex setups, compose simple functions:

```python
def extract_user(text: str) -> User:
    return client.chat.completions.create(
        model="gpt-4",
        response_model=User,
        messages=[{"role": "user", "content": text}]
    )

def validate_users(users: List[User]) -> List[User]:
    return [u for u in users if u.age > 0]

# Compose naturally
users = [extract_user(text) for text in documents]
valid_users = validate_users(users)
```

### 5. Progressive Enhancement

Start simple, add complexity only when needed:

```python
# Level 1: Basic extraction
user = client.chat.completions.create(
    response_model=User,
    messages=[...]
)

# Level 2: Add validation when needed
class User(BaseModel):
    name: str
    age: int
    
    @field_validator('age')
    def positive_age(cls, v):
        if v <= 0:
            raise ValueError('Age must be positive')
        return v

# Level 3: Add retries if validation fails
user = client.chat.completions.create(
    response_model=User,
    max_retries=3,
    messages=[...]
)

# Level 4: Stream if needed
for partial in client.chat.completions.create(
    response_model=Partial[User],
    stream=True,
    messages=[...]
):
    print(partial)
```

## What We DON'T Do

### No Prompt Engineering for You

We don't write clever prompts or "optimize" your requests. You know your domain better than we do. We just ensure the LLM returns valid data structures.

### No New Abstractions

No `Agent`, `Chain`, `Tool`, or `Workflow` classes. These are your domain concepts - implement them however makes sense for your application.

### No Hidden Magic

Every Instructor behavior can be understood by reading a single function. We prefer explicit over clever.

## The Result

By following these principles, Instructor remains:

- **Small**: Core functionality in <1000 lines of code
- **Fast**: Negligible overhead over raw API calls  
- **Reliable**: Fewer abstractions = fewer bugs
- **Learnable**: If you know Pydantic, you know Instructor

## In Practice

This philosophy means you write code like this:

```python
from pydantic import BaseModel
from typing import List
import instructor

# Your domain models - not ours
class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

class Order(BaseModel):
    products: List[Product]
    total: float

# Your business logic - not ours  
def process_order(image_path: str) -> Order:
    client = instructor.from_provider("openai/gpt-4-vision")
    
    return client.chat.completions.create(
        response_model=Order,
        messages=[{
            "role": "user",
            "content": [
                "Extract order details from this receipt",
                {"type": "image_url", "image_url": {"url": image_path}}
            ]
        }]
    )

# That's it. No framework, just functions.
```

This is the Instructor way: **Your code, your models, your logic.** We just make sure the LLM plays nice with your data.

---

> "Simplicity is the ultimate sophistication." - Leonardo da Vinci

The best code is no code. The second best is code that does exactly what it says, nothing more, nothing less. That's Instructor.