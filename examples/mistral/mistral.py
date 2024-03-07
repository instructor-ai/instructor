from pydantic import BaseModel
from mistralai.client import MistralClient
from instructor.patch import patch
from instructor.function_calls import Mode


class UserDetails(BaseModel):
    name: str
    age: int


# enables `response_model` in chat call
client = MistralClient()
patched_chat = patch(create=client.chat, mode=Mode.MISTRAL_TOOLS)

resp = patched_chat(
    model="mistral-large-latest",
    response_model=UserDetails,
    messages=[
        {
            "role": "user",
            "content": f'Extract the following entities: "Jason is 20"',
        },
    ],
)
print(resp)
