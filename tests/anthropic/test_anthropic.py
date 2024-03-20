import pytest
import anthropic
import instructor
from pydantic import BaseModel
from typing import List

create = instructor.patch(
    create=anthropic.Anthropic().messages.create, mode=instructor.Mode.ANTHROPIC_TOOLS
)

@pytest.mark.skip
def test_anthropic():
    class Properties(BaseModel):
        name: str
        value: List[str]

    class User(BaseModel):
        name: str
        age: int
        properties: List[Properties]

    resp = create(
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

    assert isinstance(resp, User)
