from litellm import completion, provider_list
from pydantic import BaseModel

import instructor
from instructor.patch import wrap_chatcompletion

completion = wrap_chatcompletion(func=completion, mode=instructor.Mode.MD_JSON)


class UserExtract(BaseModel):
    name: str
    age: int


user = completion(
    model="ollama/mistral",
    response_model=UserExtract,
    messages=[
        {
            "role": "system",
            "content": "You are a JSON Output system, only return valid JSON. YOU CAN ONLY RETURN WITH JSON NO TALKING",
        },
        {
            "role": "user",
            "content": "Extract `My name is Jason and I am 25 years old` into JSON",
        },
    ],
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"
assert user.name.lower() == "jason"
assert user.age == 25
assert hasattr(user, "_raw_response")
assert any(provider in user._raw_response.model for provider in provider_list)
