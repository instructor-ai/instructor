
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
    id='chatcmpl-9TzoQBCR4ljqayllBj0U1PKfjGqiL',
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
                        id='call_Q5iev31AA2vUVoAdpBapGWK5',
                        function=Function(
                            arguments='{"name":"Jason","age":25}', name='UserExtract'
                        ),
                        type='function',
                    )
                ],
            ),
        )
    ],
    created=1716936146,
    model='gpt-3.5-turbo-0125',
    object='chat.completion',
    system_fingerprint=None,
    usage=CompletionUsage(completion_tokens=9, prompt_tokens=82, total_tokens=91),
)
"""
```