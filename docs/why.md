---
description: Discover why Instructor is the simplest, most reliable way to get structured outputs from LLMs.
---

# Why use Instructor?

Every LLM provider now supports structured outputs. OpenAI has response_format. Anthropic has tool use. Google has function calling. The problem? They're all different, and you still have to handle validation, retries, and complex types yourself.

## The real problem: fragmentation

Yes, structured outputs exist. But look at what you're dealing with:

```python
# OpenAI's way
response = openai.chat.completions.create(
    model="gpt-4",
    response_format={"type": "json_schema", "json_schema": {...}},
    messages=[...]
)

# Anthropic's way
response = anthropic.messages.create(
    model="claude-3",
    tools=[{"name": "get_user", "input_schema": {...}}],
    messages=[...]
)

# Google's way
response = genai.generate_content(
    model="gemini-pro",
    generation_config={"response_mime_type": "application/json"},
    contents=[...]
)

# Mistral's way, Cohere's way, Groq's way...
# Each one different. Each one incomplete.
```

## What providers don't tell you

### 1. No validation beyond "valid JSON"

```python
# This is valid JSON to OpenAI:
{
    "name": "John",
    "age": "twenty-five",  # Wrong type
    "email": "not-an-email"  # Invalid format
}

# This passes their structured output:
{
    "users": null  # Where's the data?
}
```

### 2. No automatic retries

When validation fails (and it will), you're on your own:

```python
# Without Instructor: DIY retry logic
for attempt in range(3):
    response = openai.chat.completions.create(...)
    try:
        data = json.loads(response)
        # Manual validation
        if not isinstance(data.get('age'), int):
            raise ValueError("Age must be integer")
        break
    except Exception as e:
        if attempt == 2:
            raise
        # How do you tell the LLM what went wrong?
        messages.append({"role": "system", "content": f"Error: {e}"})
```

### 3. No streaming support

Want to show progress? Too bad:

```python
# Providers' structured output = all or nothing
response = openai.chat.completions.create(
    response_format={"type": "json_schema", ...},
    stream=True,  # Doesn't work with structured outputs!
)
```

### 4. No complex types

Nested objects? Enums? Custom validators? Good luck:

```python
# Try defining this with provider schemas:
class EmailStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNVERIFIED = "unverified"

class User(BaseModel):
    email: str
    status: EmailStatus
    
    @field_validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

# Provider response: ¯\_(ツ)_/¯
```

## The Instructor solution

One API that works everywhere, with everything you actually need:

```python
import instructor
from pydantic import BaseModel

# Define what you want (once)
class User(BaseModel):
    name: str
    age: int
    email: str

# Use with any provider (same code!)
client = instructor.from_provider("openai/gpt-4")
# or
client = instructor.from_provider("anthropic/claude-3")
# or
client = instructor.from_provider("google/gemini-pro")

# Get validated data with retries
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
    max_retries=3,  # Automatic retries with validation feedback
)

# Stream partial results
for partial in client.chat.completions.create(
    response_model=Partial[User],
    messages=[{"role": "user", "content": "..."}],
    stream=True,  # Actually works!
):
    print(partial)  # See results as they arrive
```

## What you get with Instructor

### Actual validation
```python
from pydantic import field_validator, EmailStr
from typing import List

class User(BaseModel):
    name: str
    age: int
    email: EmailStr  # Real email validation
    tags: List[str]
    
    @field_validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Invalid age')
        return v

# Instructor ensures you get valid data or clear errors
```

### Smart retries that work
```python
# Instructor automatically:
# 1. Catches validation errors
# 2. Feeds them back to the LLM
# 3. Retries with exponential backoff
# 4. Returns valid data or raises clear exception

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
    max_retries=3,
)
```

### Streaming that's actually useful
```python
# Show progress in real-time
for partial in client.chat.completions.create(
    response_model=Partial[Recipe],
    stream=True,
    messages=[{"role": "user", "content": "..."}],
):
    update_ui(partial)  # Update as data arrives
```

### Complex types that just work
```python
from enum import Enum
from datetime import datetime
from typing import Optional, Union

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str
    priority: Priority
    due_date: Optional[datetime]
    assignee: Union[User, str]
    subtasks: List['Task']

# Instructor handles all of this automatically
```

## The cost of rolling your own

**Time spent:**
- 2-3 days implementing validation and retries
- 1-2 days per provider integration
- Ongoing maintenance as APIs change
- Debugging "works sometimes" issues

**Bugs you'll ship:**
- Silent data corruption from bad types
- Crashes from malformed responses
- Inconsistent behavior across providers
- No visibility into what went wrong

**What you'll build anyway:**
- A worse version of Instructor
- Provider-specific spaghetti code
- Half-baked retry logic
- Manual validation everywhere

## Real developer feedback

From our GitHub and Discord:

- **"I was building this exact thing when I found Instructor"**
- **"Saved me weeks of work across multiple providers"**
- **"The validation alone is worth it"**
- **"Finally, structured outputs that actually work"**

## Start using Instructor now

```bash
pip install instructor
```

In 30 seconds:
```python
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

client = instructor.from_provider("openai/gpt-4")
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)

print(user)  # User(name='John', age=25)
```

## When to use provider-specific APIs

Let's be honest - you might not need Instructor if:

- You only use one provider and always will
- You don't need validation or retries
- You enjoy writing JSON schemas by hand
- Your data is always perfectly formatted

For everyone else building real applications, Instructor is the obvious choice.

[Get Started →](../index.md#start-in-30-seconds){ .md-button .md-button--primary }