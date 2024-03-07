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
"""
ChatCompletion(
    id='chatcmpl-8zpltT9vXJdO5OE3AfDsOhAUr911A',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content=None,
                role='assistant',
                function_call=None,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id='call_vXI3foz7jqlzFILU9pwuYJZB',
                        function=Function(
                            arguments='{"name":"Jason","age":25}', name='UserExtract'
                        ),
                        type='function',
                    )
                ],
            ),
        )
    ],
    created=1709747709,
    model='gpt-3.5-turbo-0125',
    object='chat.completion',
    system_fingerprint='fp_2b778c6b35',
    usage=CompletionUsage(completion_tokens=9, prompt_tokens=82, total_tokens=91),
)
"""
```

!!! tip "Accessing tokens usage"

    This is the recommended way to access the tokens usage, since it is a pydantic model you can use any of the pydantic model methods on it. For example, you can access the `total_tokens` by doing `user._raw_response.usage.total_tokens`. Note that this also includes the tokens used during any previous unsuccessful attempts.

    In the future, we may add additional hooks to the `raw_response` to make it easier to access the tokens usage.
