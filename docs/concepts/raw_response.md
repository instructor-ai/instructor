
# Creating a model with completions

In instructor>1.0.0 we have a custom client, if you wish to use the raw response you can do the following

```python
import instructor

from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())


class UserExtract(BaseModel):
    name: str
    age: int


user, completion = client.chat.completions.create_with_completion(
    model="gpt-3.5-turbo",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(user)
#> name='Jason' age=25

print(completion)
"""
ChatCompletion(
    id='chatcmpl-98za6bFyfGSmZ90n0QfroijxvF37Q',
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
                        id='call_9TylRriehvbU1scs1H3iDSgK',
                        function=Function(
                            arguments='{"name":"Jason","age":25}', name='UserExtract'
                        ),
                        type='function',
                    )
                ],
            ),
        )
    ],
    created=1711930370,
    model='gpt-3.5-turbo-0125',
    object='chat.completion',
    system_fingerprint='fp_b28b39ffa8',
    usage=CompletionUsage(completion_tokens=9, prompt_tokens=82, total_tokens=91),
)
"""
```