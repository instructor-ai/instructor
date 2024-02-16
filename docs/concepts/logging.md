In order to see the requests made to OpenAI and the responses, you can set logging to DEBUG. This will show the requests and responses made to OpenAI. This can be useful for debugging and understanding the requests and responses made to OpenAI.

```python
import instructor
import openai
import logging

from pydantic import BaseModel

# Set logging to DEBUG
logging.basicConfig(level=logging.DEBUG)

client = instructor.patch(openai.OpenAI())


class UserDetail(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
)
```
