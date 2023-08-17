# OpenAI Schema

The `OpenAISchema` is an extension of `Pydantic.BaseModel` that offers a minimally invasive way to define schemas for OpenAI completions. It provides two main methods: `openai_schema` to generate the correct schema and `from_response` to create an instance of the class from the completion result.

## Prompt Placement

Our philosophy is to keep prompts close to the code. This is achieved by using docstrings and field descriptions to provide prompts and descriptions for your schema fields.

## Structured Extraction

You can directly use the `OpenAISchema` class in your `openai` API create calls by passing in the `openai_schema` class property and extracting the class out using the `from_response` method. This style of usage provides full control over configuration and prompting.

```python
import openai
from instructor import OpenAISchema
from pydantic import Field

class UserDetails(OpenAISchema):
    """Details of a user"""
    name: str = Field(..., description="User's full name")
    age: int

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    functions=[UserDetails.openai_schema],
    function_call={"name": UserDetails.openai_schema["name"]},
    messages=[
        {"role": "system", "content": "Extract user details from my requests"},
        {"role": "user", "content": "My name is John Doe and I'm 30 years old."},
    ],
)

user_details = UserDetails.from_response(completion)
print(user_details)  # UserDetails(name='John Doe', age=30)
```

You can also use the `@openai_schema` decorator to decorate `BaseModels`, but you may lose some type hinting as a result.

```python
import openai
from instructor import openai_schema
from pydantic import Field, BaseModel

@openai_schema
class UserDetails(BaseModel):
    """Details of a user"""
    name: str = Field(..., description="User's full name")
    age: int
```

## Code Reference

For more information about the code, including the complete API reference, please refer to the `instructor` documentation.

::: instructor.function_calls