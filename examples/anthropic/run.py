from pydantic import BaseModel
from typing import List
import anthropic
import instructor

# Patching the Anthropics client with the instructor for enhanced capabilities
anthropic_client = instructor.patch(
    create=anthropic.Anthropic().messages.create, mode=instructor.Mode.ANTHROPIC_TOOLS
)


class Properties(BaseModel):
    key: str
    value: str


class User(BaseModel):
    name: str
    age: int
    properties: List[Properties]


user_response = anthropic_client(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    max_retries=0,
    messages=[
        {
            "role": "user",
            "content": "Create a user for a model with a name, age, and properties.",
        }
    ],
    response_model=User,
)  # type: ignore

print(user_response.model_dump_json(indent=2))
