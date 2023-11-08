# Streaming and MultiTask

A common use case of structured extraction is defining a single schema class and then making another schema to create a list to do multiple extraction

```python
class User(OpenAISchema):
    name: str
    age: int

class Users(OpenAISchema):
    users: List[User]
```

Defining a task and creating a list of classes is a common enough pattern that we define a helper function `MultiTask` It procides a function to dynamically create a new class that:

1. Dynamic docstrings and class name baed on the task
2. Helper method to support streaming by collectin function_call tokens until a object back out.

## Extracting Tasks using MultiTask

By using multitask you get a very convient class with prompts and names automatically defined. You get `from_response` just like any other `OpenAISchema` you're able to extract the list of objects data you want with `MultTask.tasks`.

```python
import instructor

# Enable `response_model`
instructor.patch()

class User(BaseModel):
    name: str
    age: int

results = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    response_model=instructor.MultiTask(User)
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

Since a `MultiTask(T)` is well contrained to `tasks: List[T]` we can make assuptions on how tokens are used and provide a helper method that allows you generate tasks as the the tokens are streamed in. This currently isnt supported via the `response_model` parameter but can be used with the `functions` parameter.

Lets look at an example in action with the same class

```python hl_lines="6 26"
MultiUser = instructor.MultiTask(User)

completion = openai.ChatCompletion.create(
    model="gpt-4-0613",
    temperature=0.1,
    stream=True,
    functions=[MultiUser.openai_schema],
    function_call={"name": MultiUser.openai_schema["name"]},
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
