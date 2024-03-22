from enum import Enum
from typing import List

import anthropic
import pytest
from pydantic import BaseModel

import instructor

create = instructor.patch(create=anthropic.Anthropic().messages.create, mode=instructor.Mode.ANTHROPIC_TOOLS)


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


@pytest.mark.skip
def test_anthropic_enum():
    class ProgrammingLanguage(Enum):
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        TYPESCRIPT = "typescript"
        UNKNOWN = "unknown"
        OTHER = "other"

    class SimpleEnum(BaseModel):
        language: ProgrammingLanguage

    resp = create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "What is your favorite programming language?",
            }
        ],
        response_model=SimpleEnum,
    )  # type: ignore

    assert isinstance(resp, SimpleEnum)
