"""
Minimal OpenAI example with Instructor.

This example demonstrates the basic usage of Instructor with OpenAI's API
to extract structured data from natural language.
"""

from pydantic import BaseModel
from openai import OpenAI
import instructor


# Define a simple response model
class User(BaseModel):
    name: str
    age: int


# Patch the OpenAI client with Instructor
client = instructor.from_openai(OpenAI())

# Extract structured data from natural language
user = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Extract: John is 25 years old.",
        }
    ],
    response_model=User,
)

print(user)
# > User(name='John', age=25)
