# Structured outputs with Together AI, a complete guide w/ instructor

Together AI provides access to various open-source models through a unified API. This guide demonstrates how to use Instructor with Together AI for structured outputs.

## Installation

```bash
pip install instructor[together]
```

## Quick Start

```python
from instructor import patch
from together import Together

# Initialize and patch the Together client
client = patch(Together(api_key="your-api-key"))
```

## Simple User Example

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int
    email: str

# Synchronous example
user = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    response_model=UserInfo,
    messages=[
        {"role": "user", "content": "Extract: John Doe is 30 years old, email: john@example.com"}
    ]
)
```

## Async Implementation

```python
import asyncio
from instructor import patch
from together import AsyncTogether

async def extract_user_info():
    client = patch(AsyncTogether(api_key="your-api-key"))

    user = await client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        response_model=UserInfo,
        messages=[
            {"role": "user", "content": "Extract: John Doe is 30 years old, email: john@example.com"}
        ]
    )
    return user

# Run async function
user = asyncio.run(extract_user_info())
```

## Nested Example

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

# Extract nested information
user = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    response_model=User,
    messages=[
        {"role": "user", "content": """
        Extract: John Doe is 30 years old
        Addresses:
        - 123 Main St, New York, USA
        - 456 Park Ave, London, UK
        """}
    ]
)
```

## Streaming Support

Together AI supports streaming responses. Here's how to use it with Instructor:

### Partial Streaming Example

```python
from typing import Optional
from pydantic import BaseModel

class PartialUser(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    email: Optional[str] = None

# Stream partial responses
for partial in client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    response_model=PartialUser,
    messages=[
        {"role": "user", "content": "Extract: John Doe is 30 years old, email: john@example.com"}
    ],
    stream=True
):
    print(f"Received partial: {partial}")
```

## Iterable Example

```python
from typing import Iterator
from pydantic import BaseModel

class Comment(BaseModel):
    author: str
    content: str

def extract_comments(text: str) -> Iterator[Comment]:
    return client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        response_model=Iterator[Comment],
        messages=[
            {"role": "user", "content": text}
        ]
    )

# Use the iterator
comments = extract_comments("""
1. @john: Great post!
2. @sarah: Thanks for sharing
3. @mike: Very informative
""")

for comment in comments:
    print(f"{comment.author}: {comment.content}")
```

## Instructor Hooks

Instructor hooks can be used with Together AI to add custom validation, logging, or transformation logic:

```python
from instructor import patch
from together import Together
from instructor.hooks import add_hook

# Define a custom hook
def logging_hook(mode: str, response_model: Any, raw_response: Any, **kwargs):
    print(f"Mode: {mode}")
    print(f"Response Model: {response_model}")
    print(f"Raw Response: {raw_response}")

# Add the hook to the patched client
client = patch(Together(api_key="your-api-key"))
add_hook(logging_hook)
```

## Best Practices

1. Choose the appropriate model based on your use case
2. Implement proper error handling
3. Use type hints and validation
4. Consider using async implementations for better performance
5. Leverage Instructor hooks for debugging and monitoring

## Related Resources

- [Together AI Documentation](https://docs.together.ai)
- [Instructor Documentation](https://instructor-ai.github.io/instructor/)
- [Pydantic Documentation](https://docs.pydantic.dev)

## Updates and Compatibility

- Together AI API is actively maintained and updated
- Instructor supports the latest Together AI API version
- Regular updates ensure compatibility with new features
