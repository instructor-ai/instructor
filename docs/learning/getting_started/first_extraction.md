# Your First Extraction

This guide walks you through creating your first structured extraction with Instructor, focusing on the basics with a simple example.

## Basic Example

Let's extract a person's name and age from text:

```python
from pydantic import BaseModel
import instructor
from openai import OpenAI

# 1. Define your data model
class Person(BaseModel):
    name: str
    age: int

# 2. Set up the Instructor client
client = instructor.from_openai(OpenAI())

# 3. Extract structured data
person = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Person,
    messages=[
        {"role": "user", "content": "John Doe is 30 years old"}
    ]
)

# 4. Use the structured data
print(f"Name: {person.name}, Age: {person.age}")
# Output: Name: John Doe, Age: 30
```

## How It Works

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Define      │ -> │ Instruct LLM │ -> │ Get Typed   │
│ Structure   │    │ to Extract   │    │ Response    │
└─────────────┘    └──────────────┘    └─────────────┘
```

Let's break down each part:

### 1. Define Your Data Model

```python
class Person(BaseModel):
    name: str
    age: int
```

This tells Instructor what to extract:
- `name`: A text string
- `age`: A whole number

### 2. Set Up the Client

```python
client = instructor.from_openai(OpenAI())
```

This wraps the OpenAI client with Instructor's functionality.

### 3. Extract Structured Data

```python
person = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Person,
    messages=[
        {"role": "user", "content": "John Doe is 30 years old"}
    ]
)
```

The key parts are:
- `model`: The LLM to use
- `response_model`: Your data structure definition
- `messages`: The text to extract from

### 4. Use the Structured Data

```python
print(f"Name: {person.name}, Age: {person.age}")
```

The result is a Python object with type-safe properties.

## Adding Field Descriptions

You can add descriptions to help guide the extraction:

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
```

These descriptions help the model understand what to extract.

## Handling Missing Information

If information might be missing, make fields optional:

```python
from typing import Optional

class Person(BaseModel):
    name: str
    age: Optional[int] = None  # Now age is optional
```

## Next Steps

Now that you've created your first extraction, you can:
1. Learn about more complex [Response Models](response_models.md)
2. Set up different [Client Configurations](client_setup.md)
3. Explore [Simple Object Extraction](../patterns/simple_object.md) 