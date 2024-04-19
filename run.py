from openai import OpenAI
from instructor import from_openai
from instructor.retry import InstructorRetryException
from pydantic import BaseModel, field_validator

# Patch the OpenAI client to enable response_model
client = from_openai(OpenAI())


# Define a Pydantic model for the user details
class UserDetail(BaseModel):
    name: str
    age: int

    @field_validator("age")
    def validate_age(cls, v):
        raise ValueError("You will never succeed")


# Use the client to create a user detail
try:
    user: UserDetail = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[{"role": "user", "content": "Extract Jason is 25 years old"}],
        max_retries=3,
    )
except InstructorRetryException as e:
    print(e)
    """
    1 validation error for UserDetail
    age
        Value error, You will never succeed [type=value_error, input_value=25, input_type=int]
            For further information visit https://errors.pydantic.dev/2.7/v/value_error
    """

    print(e.n_attempts)
    # > 3

    print(e.last_completion)
    """
    ChatCompletion(id='chatcmpl-9FaHq4dL4SszLAbErGlpD3a0TYxi0', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_XidgLpIu1yfaq876L65k91RM', function=Function(arguments='{"name":"Jason","age":25}', name='UserDetail'), type='function')]))], created=1713501434, model='gpt-3.5-turbo-0125', object='chat.completion', system_fingerprint='fp_d9767fc5b9', usage=CompletionUsage(completion_tokens=27, prompt_tokens=513, total_tokens=540))
    """
