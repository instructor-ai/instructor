# Multi-task and Streaming

A common use case of structured extraction is defining a single schema class and then making another schema to create a list to do multiple extraction

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

class Users(BaseModel):
    users: List[User]
```

Defining a task and creating a list of classes is a common enough pattern that we make this convenient by making use of `Iterable[T]`. This lets us dynamically create a new class that:

1. Has dynamic docstrings and class name based on the task
2. Support streaming by collecting tokens until a task is received back out.

## Extracting Tasks using Iterable

By using `Iterable` you get a very convient class with prompts and names automatically defined:

```python
import instructor
from openai import OpenAI
from typing import Iterable
from pydantic import BaseModel

client = instructor.patch(OpenAI(), mode=instructor.function_calls.Mode.JSON)

class User(BaseModel):
    name: str
    age: int

Users = Iterable[User]

users = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    temperature=0.1,
    response_model=Users,
    stream=False,
    messages=[
        {
            "role": "user",
            "content": "Consider this data: Jason is 10 and John is 30.\
                         Correctly segment it into entitites\
                        Make sure the JSON is correct",
        },
    ],
)
for user in users:
    assert isinstance(user, User)
    print(user)

>>> name="Jason" "age"=10
>>> name="John" "age"=10
```

## Streaming Tasks

We can also generate tasks as the tokens are streamed in by defining an `Iterable[T]` type.

Lets look at an example in action with the same class

```python hl_lines="6 26"
from typing import Iterable

Users = Iterable[User]

users = client.chat.completions.create(
    model="gpt-4",
    temperature=0.1,
    stream=True,
    response_model=Users,
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

for user in users:
    assert isinstance(user, User)
    print(user)

>>> name="Jason" "age"=10
>>> name="John" "age"=10
```

This streaming is still a prototype, but should work quite well for simple schemas.
