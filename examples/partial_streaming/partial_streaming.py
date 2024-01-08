import instructor
from openai import OpenAI
from typing import Iterable, Optional, List
from pydantic import BaseModel, Field
from instructor import Partial, Maybe
import json

client = instructor.patch(OpenAI())

class Address(BaseModel):
    state: str

class User(BaseModel):
    name: str
    age: str
    address: Address

PartialUser = Partial(User)

string = "Jason. free dives. He is 25 years old. He lives in California"

users_stream = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        response_model=Iterable[PartialUser],
        messages=[
            {
                "role": "user",
                "content": f"Get only the user details for this information about the user {string}",
            },
        ],
        stream=True
    )  # type: ignore


for user in users_stream:
    print(user)


# class User(BaseModel):
#     name: str
#     age: str

# class Users(BaseModel):
#    users: List[User]
#    comments: str

# Code below throws errors unable to make Users a Partial class
# PartialUsers = Partial(Users)

# print(json.dumps(PartialUser.model_json_schema()))