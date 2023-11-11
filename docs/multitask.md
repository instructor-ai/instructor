# Streaming and MultiTask

A common use case of structured extraction is defining a single schema class and then making another schema to create a list to do multiple extraction

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

class Users(BaseModel):
    users: List[User]
```

Defining a task and creating a list of classes is a common enough pattern that we define a helper function `MultiTask` It procides a function to dynamically create a new class that:

1. Dynamic docstrings and class name baed on the task
2. Helper method to support streaming by collectin function_call tokens until a object back out.

## Extracting Tasks using MultiTask

By using multitask you get a very convient class with prompts and names automatically defined. You get `from_response` just like any other `BaseModel` you're able to extract the list of objects data you want with `MultTask.tasks`.

```python hl_lines="13"
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI())


class User(BaseModel):
    name: str
    age: int


MultiUser = instructor.MultiTask(User)

completion = client.chat.completions.create(
    model="gpt-4-0613",
    temperature=0.1,
    stream=False,
    functions=[MultiUser.openai_schema],
    function_call={"name": MultiUser.openai_schema["name"]},
    messages=[
        {
            "role": "user",
            "content": f"Consider the data below: Jason is 10 and John is 30",
        },
    ],
)
```

```json
{
  "tasks": [
    { "name": "Jason", "age": 10 },
    { "name": "John", "age": 30 }
  ]
}
```

## Streaming Tasks

Since a `MultiTask(T)` is well contrained to `tasks: List[T]` we can make assuptions on how tokens are used and provide a helper method that allows you generate tasks as the the tokens are streamed in

Lets look at an example in action with the same class

```python hl_lines="6 26"
MultiUser = instructor.MultiTask(User)

completion = client.chat.completions.create(
    model="gpt-4-0613",
    temperature=0.1,
    stream=True,
    response_model=MultiUser,
    messages=[
        {
            "role": "system",
            "content": "You are a perfect entity extraction system",
        },
        {
            "role": "user",
            "content": (
                f"Consider the data below:\n{input}"
                "Correctly segment it into entitites"
                "Make sure the JSON is correct"
            ),
        },
    ],
    max_tokens=1000,
)

for user in MultiUser.from_streaming_response(completion):
    assert isinstance(user, User)
    print(user)

>>> name="Jason" "age"=10
>>> name="John" "age"=10
```

This streaming is still a prototype, but should work quite well for simple schemas.
