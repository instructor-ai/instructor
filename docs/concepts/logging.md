In order to see the requests made to OpenAI and the responses, you can set logging to DEBUG. This will show the requests and responses made to OpenAI. This can be useful for debugging and understanding the requests and responses made to OpenAI. I would love some contributions that make this a lot cleaner, but for now this is the fastest way to see the prompts. 

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
)  # type: ignore

""" 
...
DEBUG:instructor:Patching `client.chat.completions.create` with mode=<Mode.TOOLS: 'tool_call'>
DEBUG:instructor:Instructor Request: mode.value='tool_call', response_model=<class '__main__.UserDetail'>, new_kwargs={'model': 'gpt-3.5-turbo', 'messages': [{'role': 'user', 'content': 'Extract Jason is 25 years old'}], 'tools': [{'type': 'function', 'function': {'name': 'UserDetail', 'description': 'Correctly extracted `UserDetail` with all the required parameters with correct types', 'parameters': {'properties': {'name': {'title': 'Name', 'type': 'string'}, 'age': {'title': 'Age', 'type': 'integer'}}, 'required': ['age', 'name'], 'type': 'object'}}}], 'tool_choice': {'type': 'function', 'function': {'name': 'UserDetail'}}}
DEBUG:instructor:max_retries: 1
...
DEBUG:instructor:Instructor Pre-Response: ChatCompletion(id='chatcmpl-8zBxMxsOqm5Sj6yeEI38PnU2r6ncC', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_E1cftF5U0zEjzIbWt3q0ZLbN', function=Function(arguments='{"name":"Jason","age":25}', name='UserDetail'), type='function')]))], created=1709594660, model='gpt-3.5-turbo-0125', object='chat.completion', system_fingerprint='fp_2b778c6b35', usage=CompletionUsage(completion_tokens=9, prompt_tokens=81, total_tokens=90))
DEBUG:httpcore.connection:close.started
DEBUG:httpcore.connection:close.complete
"""
```