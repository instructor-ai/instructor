---
draft: False
date: 2024-03-20
authors:
  - jxnl
---

# Announcing Anthropic Support

A special shoutout to [Shreya](https://twitter.com/shreyaw_) for her contributions to the anthropic support. As of now, all features are operational with the exception of streaming support.

For those eager to experiment, simply patch the client with `ANTHROPIC_JSON`, which will enable you to leverage the `anthropic` client for making requests.

```
pip install instructor[anthropic]
```

!!! warning "Missing Features"

    Just want to acknowledge that we know that we are missing partial streaming and some better re-asking support for XML. We are working on it and will have it soon.

```python
from pydantic import BaseModel
from typing import List
import anthropic
import instructor

# Patching the Anthropics client with the instructor for enhanced capabilities
anthropic_client = instructor.patch(
    create=anthropic.Anthropic().messages.create,
    mode=instructor.Mode.ANTHROPIC_JSON
)

class Properties(BaseModel):
    name: str
    value: str

class User(BaseModel):
    name: str
    age: int
    properties: List[Properties]

user_response = anthropic_client(
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
    "name": "John",
    "age": 25,
    "properties": [
        {
            "key": "favorite_color",
            "value": "blue"
        }
    ]
}
```

We're encountering challenges with deeply nested types and eagerly invite the community to test, provide feedback, and suggest necessary improvements as we enhance the anthropic client's support.