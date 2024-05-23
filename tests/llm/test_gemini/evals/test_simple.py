import google.generativeai as genai
from pydantic import BaseModel, field_validator

import instructor

client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest",
    ),
    mode=instructor.Mode.GEMINI_JSON,
)


def test_simple():
    class User(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_is_uppercase(cls, v: str):
            assert v.isupper(), "Name must be uppercase, please fix"
            return v

    resp = client.messages.create(
        messages=[
            {
                "role": "user",
                "content": "Extract John is 18 years old.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert resp.name == "JOHN"  # due to validation
    assert resp.age == 18
