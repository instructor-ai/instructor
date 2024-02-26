from pydantic import BaseModel, Field
from typing import Optional
from mistralai.client import MistralClient
from instructor.patch import patch
from instructor.function_calls import Mode

client = MistralClient()
new_chat = patch(create=client.chat, mode=Mode.MIST_TOOLS)


class UserDetail(BaseModel):
    age: Optional[int] = None
    name: Optional[str] = None
    role: Optional[str] = None


def get_user_detail(string) -> UserDetail:
    return new_chat(
        model="mistral-large-latest",
        response_model=UserDetail,
        messages=[
            {
                "role": "user",
                "content": f"Get user details for {string}",
            },
        ],
    )  # type: ignore


user = get_user_detail("Jason is 25 years old")
print(user.model_dump_json(indent=2))
"""
{
  "age": 25,
  "name": "Jason",
  "role": null
}
"""

user = get_user_detail("Jason is a 25 years old scientist")
print(user.model_dump_json(indent=2))
"""
{
  "age": 25,
  "name": "Jason",
  "role": "scientist"
}
"""

user = get_user_detail("User not found")
print(user.model_dump_json(indent=2))
"""
{
  "age": null,
  "name": null,
  "role": null
}
"""