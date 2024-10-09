import instructor
import google.generativeai as genai
from pydantic import BaseModel
from .util import models, modes


class User(BaseModel):
    first_name: str
    age: int


import pytest
from itertools import product


@pytest.mark.parametrize("model, mode, is_list", product(models, modes, [True, False]))
def test_format_string(model: str, mode: instructor.Mode, is_list: bool):
    client = instructor.from_gemini(
        client=genai.GenerativeModel(
            model_name=model,
            system_instruction="You are a helpful assistant that excels at extracting user information.",
        ),
        mode=mode,
    )

    content = (
        ["Extract {{name}} is {{age}} years old."]
        if is_list
        else "Extract {{name}} is {{age}} years old."
    )

    # note that client.chat.completions.create will also work
    resp = client.messages.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        response_model=User,
        context={"name": "Jason", "age": 25},
    )

    assert isinstance(resp, User)
    assert resp.first_name == "Jason"
    assert resp.age == 25
