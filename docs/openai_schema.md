# OpenAI Schema

We offer a minimally invasive extention of `Pydantic.BaseModel` named `OpenAISchema`. It only has two methods, one to generate the correct schema, and one to produce the class from the completion.

!!! note "Where does the prompt go?"
    Our philosphy is that the prompt should live beside the code. Prompting is done via dostrings and field descriptions which allows you to colocate prompts with  your schema.

## Structured Extraction

You can directly use the class in your `openai` create calls by passing in the classes `openai_schema` and extract the class out with `from_completion`.

With this style of usage you get as close to the api call as possible giving you full control over configuration and prompting.

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

!!! tip "Using the decorator"
    You can also use `@openai_schema` to decorate `BaseModels`, however, you'll lose some type hinting as a result.

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
