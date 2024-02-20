The easiest way to get usage for non streaming requests is to access the raw response.

```python
import instructor

from openai import OpenAI
from pydantic import BaseModel

client = instructor.patch(OpenAI())


class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(user._raw_response.usage)
#> CompletionUsage(completion_tokens=9, prompt_tokens=82, total_tokens=91)
```
