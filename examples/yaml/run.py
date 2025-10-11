"""Simple YAML mode example with structured outputs."""

import instructor
from openai import OpenAI
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Initialize the client with YAML mode
client = instructor.from_openai(OpenAI(), mode=instructor.Mode.YAML)

# Extract structured data using YAML mode
user = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Create a user named John who is 30 years old"}
    ],
    response_model=User,
)

print(f"Name: {user.name}")
print(f"Age: {user.age}")

# Expected output:
# Name: John
# Age: 30
