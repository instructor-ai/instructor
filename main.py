from instructor import from_openai
from openai import OpenAI
from instructor import Mode
from pydantic import BaseModel
import os

client = from_openai(
    OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/",
    ),
    mode=Mode.MD_JSON,
)


class User(BaseModel):
    name: str
    age: int


resp = client.chat.completions.create_iterable(
    model="gemini-1.5-flash",
    messages=[
        {
            "role": "user",
            "content": "Generate 10 random users",
        }
    ],
    response_model=User,
)

for r in resp:
    print(r)
