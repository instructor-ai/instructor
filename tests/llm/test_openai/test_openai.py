import instructor
from pydantic import BaseModel


def test_openai():
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_provider("openai/gpt-4o-mini")
    response = client.create(
        messages=[
            {
                "role": "user",
                "content": "Extract the name and age from the following text: 'My name is Jason and I am 25 years old.'",
            }
        ],
        response_model=User,
    )
    assert response.age == 25
