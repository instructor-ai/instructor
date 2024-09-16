from itertools import product
import pytest
import instructor
import google.generativeai as genai
from pydantic import BaseModel
from .util import models, modes


class User(BaseModel):
    name: str
    age: int


class UserList(BaseModel):
    items: list[User]


def test_list_of_strings():
    client = instructor.from_gemini(
        genai.GenerativeModel("gemini-1.5-flash-latest"),
        mode=instructor.Mode.GEMINI_JSON,
    )

    content = [
        "Extract a list of users from the following text",
        "Jason is 25 years old",
        "Elizabeth is 12 years old",
        "Chris is 27 years old",
    ]

    result = client.chat.completions.create(
        response_model=UserList,
        messages=[
            {"role": "user", "content": content},
        ],
    )

    assert isinstance(result, UserList), "Result should be an instance of UserList"
    assert isinstance(result.items, list), "items should be a list"
    assert len(result.items) == 3, "List should contain 3 items"

    names = [item.name.upper() for item in result.items]
    assert "JASON" in names, "'JASON' should be in the list"
    assert "ELIZABETH" in names, "'ELIZABETH' should be in the list"
    assert "CHRIS" in names, "'CHRIS' should be in the list"
