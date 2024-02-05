# Patching

Instructor enhances client functionality with three new keywords for backwards compatibility. This allows use of the enhanced client as usual, with structured output benefits.

- `response_model`: Defines the response type for `chat.completions.create`.
- `max_retries`: Determines retry attempts for failed `chat.completions.create` validations.
- `validation_context`: Provides extra context to the validation process.

There are three methods for structured output:

1. **Function Calling**: The primary method. Use this for stability and testing.
2. **Tool Calling**: Useful in specific scenarios; lacks the reasking feature of OpenAI's tool calling API.
3. **JSON Mode**: Offers closer adherence to JSON but with more potential validation errors. Suitable for specific non-function calling clients.

## Function Calling

```python
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI(), mode=instructor.Mode.FUNCTIONS)
```

## Tool Calling

```python
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI(), mode=instructor.Mode.TOOLS)
```

## JSON Mode

```python
import instructor
from instructor import Mode
from openai import OpenAI

client = instructor.patch(OpenAI(), mode=Mode.JSON)
```

## Markdown JSON Mode

!!! warning "Experimental"

    This is not recommended, and may not be supported in the future, this is just left to support vision models.

```python
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI(), mode=instructor.Mode.MD_JSON)
```

### Schema Integration

In JSON Mode, the schema is part of the system message:

```python
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI())


class UserExtract(instructor.OpenAISchema):
    name: str
    age: int


response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": f"Match your response to this json_schema: \n{UserExtract.model_json_schema()['properties']}",
        },
        {
            "role": "user",
            "content": "Extract jason is 25 years old",
        },
    ],
)
user = UserExtract.from_response(response, mode=instructor.Mode.JSON)
print(user)
#> name='Jason' age=25
```
