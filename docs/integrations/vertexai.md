# Structured outputs with Vertex AI, a complete guide w/ instructor

Vertex AI is Google Cloud's unified ML platform that provides access to various AI models. This guide demonstrates how to use Instructor with Vertex AI for structured outputs.

## Installation

```bash
pip install instructor[vertexai]
```

## Quick Start

```python
from instructor import patch
from vertexai.language_models import ChatModel

# Initialize and patch the Vertex AI client
chat_model = patch(ChatModel.from_pretrained("chat-bison@002"))
```

## Simple User Example

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int
    email: str

# Synchronous example
chat = chat_model.start_chat()
user = chat.send_message(
    "Extract: John Doe is 30 years old, email: john@example.com",
    response_model=UserInfo
)
```

## Async Implementation

```python
import asyncio
from instructor import patch
from vertexai.language_models import ChatModel

async def extract_user_info():
    chat_model = patch(ChatModel.from_pretrained("chat-bison@002"))
    chat = chat_model.start_chat()

    user = await chat.send_message_async(
        "Extract: John Doe is 30 years old, email: john@example.com",
        response_model=UserInfo
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
chat = chat_model.start_chat()
user = chat.send_message(
    """
    Extract: John Doe is 30 years old
    Addresses:
    - 123 Main St, New York, USA
    - 456 Park Ave, London, UK
    """,
    response_model=User
)
```

## Streaming Support

Vertex AI supports streaming responses. Here's how to use it with Instructor:

### Partial Streaming Example

```python
from typing import Optional
from pydantic import BaseModel

class PartialUser(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    email: Optional[str] = None

# Stream partial responses
chat = chat_model.start_chat()
for partial in chat.send_message(
    "Extract: John Doe is 30 years old, email: john@example.com",
    response_model=PartialUser,
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
    chat = chat_model.start_chat()
    return chat.send_message(
        text,
        response_model=Iterator[Comment]
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

Instructor hooks can be used with Vertex AI to add custom validation, logging, or transformation logic:

```python
from instructor import patch
from vertexai.language_models import ChatModel
from instructor.hooks import add_hook

# Define a custom hook
def logging_hook(mode: str, response_model: Any, raw_response: Any, **kwargs):
    print(f"Mode: {mode}")
    print(f"Response Model: {response_model}")
    print(f"Raw Response: {raw_response}")

# Add the hook to the patched model
chat_model = patch(ChatModel.from_pretrained("chat-bison@002"))
add_hook(logging_hook)
```

## Best Practices

1. Choose the appropriate model based on your use case
2. Implement proper error handling
3. Use type hints and validation
4. Consider using async implementations for better performance
5. Leverage Instructor hooks for debugging and monitoring

## Related Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Instructor Documentation](https://instructor-ai.github.io/instructor/)
- [Pydantic Documentation](https://docs.pydantic.dev)

## Updates and Compatibility

- Vertex AI API is actively maintained and updated
- Instructor supports the latest Vertex AI API version
- Regular updates ensure compatibility with new features
