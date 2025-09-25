---
title: Handling Non-Streaming Requests in OpenAI with Usage Tracking
description: Learn how to manage non-streaming requests in OpenAI, track token usage, and handle exceptions with Python.
---

## Accessing Usage Information

With instructor, you can access token usage information directly from the response model using the convenient `usage` property:

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4.1-mini")

class UserExtract(BaseModel):
    name: str
    age: int

# NEW: Direct access to usage information
user = client.chat.completions.create(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

# Access usage information directly from the model
print(user.usage)
"""
CompletionUsage(
    completion_tokens=10,
    prompt_tokens=82,
    total_tokens=92,
    completion_tokens_details=CompletionTokensDetails(
        audio_tokens=0, reasoning_tokens=0
    ),
    prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
)
"""

# You can also access individual fields
print(f"Total tokens: {user.usage.total_tokens}")        # 92
print(f"Prompt tokens: {user.usage.prompt_tokens}")      # 82  
print(f"Completion tokens: {user.usage.completion_tokens}") # 10
```

### Works with All Providers

The usage property works consistently across all supported providers, including Anthropic:

```python
import instructor
from pydantic import BaseModel

# Works with Anthropic
client = instructor.from_provider("anthropic/claude-3-haiku")

class UserExtract(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

# Access Anthropic usage information
print(user.usage.input_tokens)   # Input tokens (similar to prompt_tokens)
print(user.usage.output_tokens)  # Output tokens (similar to completion_tokens)
```

## Alternative Methods

### Using create_with_completion

You can also get usage for non-streaming requests by accessing the raw response using `create_with_completion`:

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4.1-mini")

class UserExtract(BaseModel):
    name: str
    age: int

user, completion = client.chat.completions.create_with_completion(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(completion.usage)
"""
CompletionUsage(
    completion_tokens=10,
    prompt_tokens=82,
    total_tokens=92,
    completion_tokens_details=CompletionTokensDetails(
        audio_tokens=0, reasoning_tokens=0
    ),
    prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
)
"""
```

### Using Raw Response

For advanced use cases, you can still access the raw response:

```python
user = client.chat.completions.create(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

# Access raw response usage (equivalent to user.usage)
print(user._raw_response.usage)
```

You can catch an IncompleteOutputException whenever the context length is exceeded and react accordingly, such as by trimming your prompt by the number of exceeding tokens.

```python
from instructor.core.exceptions import IncompleteOutputException
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4.1-mini")


class UserExtract(BaseModel):
    name: str
    age: int


try:
    client.chat.completions.create_with_completion(
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
except IncompleteOutputException as e:
    token_count = e.last_completion.usage.total_tokens  # type: ignore
    # your logic here
```
