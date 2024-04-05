# Anthropic 

Now that we have a [Anthropic](https://www.anthropic.com/) client, we can use it with the `instructor` client to make requests.

```
pip install anthropic
```

```python
from pydantic import BaseModel
from typing import List
import anthropic
import instructor

# Patching the Anthropics client with the instructor for enhanced capabilities
client = instructor.from_anthropic(
    anthropic.Anthropic(),
)


class Properties(BaseModel):
    name: str
    value: str


class User(BaseModel):
    name: str
    age: int
    properties: List[Properties]


# client.messages.create will also work due to the instructor client
user_response = client.chat.completions.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    max_retries=0,
    messages=[
        {
            "role": "user",
            "content": "Create a user for a model with a name, age, and properties.",
        }
    ],
    response_model=User,
)  # type: ignore

print(user_response.model_dump_json(indent=2))
"""
{
  "name": "John Doe",
  "age": 35,
  "properties": [
    {
      "name": "Hair Color",
      "value": "Brown"
    },
    {
      "name": "Height",
      "value": "6'2\""
    },
    {
      "name": "Occupation",
      "value": "Software Engineer"
    }
  ]
}
"""
```

We're encountering challenges with deeply nested types and eagerly invite the community to test, provide feedback, and suggest necessary improvements as we enhance the anthropic client's support.