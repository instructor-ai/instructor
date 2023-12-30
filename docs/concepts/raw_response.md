Often times not only do you want the base model but may also want the original response from the API. You can do this by retrieving the `raw_response`, since the `raw_response` is also a pydantic model, you can use any of the pydantic model methods on it.

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

print(user._raw_response)
```

**Output:**

```python
{
  "id": "chatcmpl-8bHUPGZc9vAXBraJlebf8ciz4AMuh",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": null,
        "role": "assistant",
        "function_call": {
          "arguments": "{\n  \"name\": \"Jason\",\n  \"age\": 25\n}",
          "name": "UserExtract"
        },
        "tool_calls": null
      },
      "logprobs": null
    }
  ],
  "created": 1703896057,
  "model": "gpt-3.5-turbo-0613",
  "object": "chat.completion",
  "system_fingerprint": null,
  "usage": {
    "completion_tokens": 16,
    "prompt_tokens": 73,
    "total_tokens": 89
  }
}

```
