from litellm import completion
from pydantic import BaseModel

import instructor
from instructor.patch import wrap_chatcompletion

completion = wrap_chatcompletion(completion, mode=instructor.Mode.MD_JSON)


class User(BaseModel):
    name: str
    age: int


response = completion(
    model="ollama/mistral",
    response_model=User,
    messages=[
        {
            "role": "system",
            "content": "You are a JSON Output system, only return valid JSON. YOU CAN ONLY RETURN WITH JSON NO TALKING",
        },
        {
            "role": "user",
            "content": "Extract `My name is John and I am 25 years old` into JSON",
        },
    ],
    api_base="http://localhost:11434",
)

assert response.name == "John", "Name is not John"
assert response.age == 25, "Age is not 25"
