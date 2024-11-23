# Working with Union Types in Instructor

This guide explains how to work with union types in Instructor, allowing you to handle multiple possible response types from language models.

## Basic Union Types

Union types let you specify that a field can be one of several types:

```python
from typing import Union
from pydantic import BaseModel


class Response(BaseModel):
    value: Union[str, int]  # Can be either string or integer
```

## Discriminated Unions

Use discriminated unions to handle different response types:

```python
from typing import Literal, Union
from pydantic import BaseModel


class UserQuery(BaseModel):
    type: Literal["user"]
    username: str


class SystemQuery(BaseModel):
    type: Literal["system"]
    command: str


Query = Union[UserQuery, SystemQuery]

# Usage with Instructor
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Query,
    messages=[{"role": "user", "content": "Parse: user lookup jsmith"}],
)
```

## Optional Fields

Combine Union with Optional for nullable fields:

```python
from typing import Optional
from pydantic import BaseModel


class User(BaseModel):
    name: str
    email: Optional[str] = None  # Same as Union[str, None]
```

## Best Practices

1. **Type Hints**: Use proper type hints for clarity
2. **Discriminators**: Add discriminator fields for complex unions
3. **Validation**: Add validators for union fields
4. **Documentation**: Document expected types clearly

## Common Patterns

### Multiple Response Types
```python
from typing import Union, Literal
from pydantic import BaseModel


class SuccessResponse(BaseModel):
    status: Literal["success"]
    data: dict


class ErrorResponse(BaseModel):
    status: Literal["error"]
    message: str


Response = Union[SuccessResponse, ErrorResponse]
```

### Nested Unions
```python
from typing import Union, List
from pydantic import BaseModel


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageContent(BaseModel):
    type: Literal["image"]
    url: str


class Message(BaseModel):
    content: List[Union[TextContent, ImageContent]]
```

## Integration with Instructor

### Validation with Unions
```python
from instructor import patch
from openai import OpenAI

client = patch(OpenAI())


def validate_response(response: Response) -> bool:
    if isinstance(response, ErrorResponse):
        return len(response.message) > 0
    return True


result = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Response,
    validation_hook=validate_response,
    messages=[{"role": "user", "content": "Process this request"}],
)
```

### Streaming with Unions
```python
def stream_content():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Message,
        stream=True,
        messages=[{"role": "user", "content": "Generate mixed content"}],
    )
    for partial in response:
        if partial.content:
            for item in partial.content:
                if isinstance(item, TextContent):
                    print(f"Text: {item.text}")
                elif isinstance(item, ImageContent):
                    print(f"Image: {item.url}")
```

## Error Handling

Handle union type validation errors:

```python
from pydantic import ValidationError

try:
    response = Response(status="invalid", data={"key": "value"})  # Invalid status
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Type Checking

Use isinstance() for runtime type checking:

```python
def process_response(response: Response):
    if isinstance(response, SuccessResponse):
        # Handle success case
        process_data(response.data)
    elif isinstance(response, ErrorResponse):
        # Handle error case
        log_error(response.message)
```

For more information about union types, check out the [Pydantic documentation on unions](https://docs.pydantic.dev/latest/concepts/types/#unions).
