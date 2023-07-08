# OpenAI Schema

The most generic helper is a light weight extention of Pydantic's BaseModel `OpenAISchema`.
It has a method to help you produce the schema and parse the result of function calls

This library is moreso a list of examples and a helper class so I'll keep the example as just structured extraction.

## Where does the prompts go?

Instead of defining your prompts in the messages the prompts you would usually use are now defined as part of the dostring of your class and the field descriptions. This is nice since it allows you to colocate the schema with the class you use to represent the structure.

## Structured Extraction

```python
import openai
from openai_function_call import OpenAISchema

from pydantic import Field

class UserDetails(OpenAISchema):
    """Details of a user"""
    name: str = Field(..., description="users's full name")
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
print(user_details)  # name="John Doe", age=30
```

## Using the decorator

You can also use a decorator but i recommend the class since you get nice autocompletes with VSCode

```python
import openai
from openai_function_call import openai_schema

from pydantic import Field, BaseModel

@openai_schema
class UserDetails(BaseModel):
    """Details of a user"""
    name: str = Field(..., description="users's full name")
    age: int
```

## Code Reference

::: openai_function_call.function_calls
