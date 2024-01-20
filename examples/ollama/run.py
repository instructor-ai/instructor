from litellm import completion, provider_list
from pydantic import BaseModel

import instructor
from instructor.patch import wrap_chatcompletion

completion = wrap_chatcompletion(func=completion, mode=instructor.Mode.MD_JSON)


class UserExtract(BaseModel):
    name: str
    age: int


user = completion(
    model="ollama/llama2",
    response_model=UserExtract,
    messages=[
        {
            "role": "system",
            "content": "You are a JSON extractor. Please extract the following JSON, No Talk. You must return JSON right after the Codeblock",
        },
        {
            "role": "user",
            "content": "Extract `My name is Jason and I am 25 years old`",
        },
    ],
)

print(user.model_dump_json(indent=2))
assert isinstance(user, UserExtract), "Should be instance of UserExtract"
assert user.name.lower() == "jason"
assert user.age == 25
assert hasattr(user, "_raw_response")
assert any(provider in user._raw_response.model for provider in provider_list)
