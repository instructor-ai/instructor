# TypedDicts

We also support typed dicts.

```python
from typing_extensions import TypedDict
from openai import OpenAI
import instructor


class User(TypedDict):
    name: str
    age: int


client = instructor.from_openai(OpenAI())


response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Timothy is a man from New York who is turning 32 this year",
        }
    ],
)
```