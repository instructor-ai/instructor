import instructor
import google.generativeai as genai
from pydantic import BaseModel

roles = [
    "system",
    "user",
    "assistant",
]


def test_roles():
    client = instructor.from_gemini(
        client=genai.GenerativeModel(
            model_name="models/gemini-1.5-flash-latest",
        ),
        mode=instructor.Mode.GEMINI_JSON,
    )

    class Description(BaseModel):
        description: str

    for role in roles:
        resp = client.create(
            response_model=Description,
            messages=[
                {
                    "role": role,
                    "content": "Describe what a sunset in the desert looks like.",
                },
                {
                    "role": "user",
                    "content": "Please adhere to the instructions",
                },
            ],
        )

        assert isinstance(resp, Description)
