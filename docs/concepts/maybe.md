# Handling Missing Data with `Maybe`

In this post, we will demonstrate how to use the `Maybe` pattern to manage missing data and employ pattern matching to handle errors in a structured manner.

## What is `Maybe`?

The `Maybe` pattern is a concept in functional programming used for error handling. Instead of raising exceptions or returning `None`, you can use a `Maybe` type to encapsulate both the result and potential errors. This pattern is particularly useful when making OpenAI API calls, as providing language models with an escape mechanism effectively reduces hallucinations. Consequently, we can construct a prompt that closely resembles regular programming.

Towards the end, we will demonstrate how to use `Maybe` instances in pattern matching, which offers an excellent approach for handling errors in a structured manner.

## Defining the Model

Using Pydantic, we'll first define the `UserDetail` and `MaybeUser` classes.

```python
from pydantic import BaseModel, Field, Optional

class UserDetail(BaseModel):
    age: int
    name: str
    role: Optional[str] = Field(default=None)

class MaybeUser(BaseModel):
    result: Optional[UserDetail] = Field(default=None)
    error: bool = Field(default=False)
    message: Optional[str] = Field(default=None)

    def __bool__(self):
        return self.result is not None
```

Notice that `MaybeUser` has a `result` field that is an optional `UserDetail` instance where the extracted data will be stored. The `error` field is a boolean that indicates whether an error occurred, and the `message` field is an optional string that contains the error message.

## Defining the function

Once we have the model defined, we can create a function that uses the `Maybe` pattern to extract the data.

```python
import random
import instructor
from openai import OpenAI
from typing import Optional

# This enables the `response_model` keyword
client = instructor.patch(OpenAI())

def extract(content: str) -> MaybeUser:
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=MaybeUser,
        messages=[
            {"role": "user", "content": f"Extract `{content}`"},
        ],
    )

user1 = extract("Jason is a 25-year-old scientist")
# output:
{
  "result": {
    "age": 25,
    "name": "Jason",
    "role": "scientist"
  },
  "error": false,
  "message": null
}

user2 = extract("Unknown user")
# output:
{
  "result": null,
  "error": true,
  "message": "User not found"
}
```

As you can see, when the data is extracted successfully, the `result` field contains the `UserDetail` instance. When an error occurs, the `error` field is set to `True`, and the `message` field contains the error message.

## Handle the result

There are a few ways we can handle the result. Normally, we can just access the individual fields.

```python
def process_user_detail(maybe_user: MaybeUser):
  if not maybe_user.error:
      user = maybe_user.result
      print(f"User {user.name} is {user.age} years old")
  else:
      print(f"Not found: {user1.message}")
```

## Pattern Matching

We can also use pattern matching to handle the result. This is a great way to handle errors in a structured way.

```python
def process_user_detail(maybe_user: MaybeUser):
    match maybe_user:
        case MaybeUser(error=True, message=msg):
            print(f"Error: {msg}")
        case MaybeUser(result=user_detail) if user_detail:
            assert isinstance(user_detail, UserDetail)
            print(f"User {user_detail.name} is {user_detail.age} years old")
        case _:
            print("Unknown error")
```

If you want to learn more about pattern matching, check out Pydantic's docs on [Structural Pattern Matching](https://docs.pydantic.dev/latest/concepts/models/#structural-pattern-matching)
