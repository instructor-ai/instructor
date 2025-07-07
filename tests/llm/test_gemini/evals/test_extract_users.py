import pytest
from pydantic import BaseModel
import instructor


class UserDetails(BaseModel):
    name: str
    age: int


# Lists for models, test data, and modes
test_data = [
    ("Jason is 10", "Jason", 10),
    ("Alice is 25", "Alice", 25),
    ("Bob is 35", "Bob", 35),
]


@pytest.mark.parametrize("data", test_data)
def test_extract(data):
    sample_data, expected_name, expected_age = data

    client = instructor.from_provider("google/gemini-2.0-flash-exp")

    # Calling the extract function with the provided model, sample data, and mode
    response = client.chat.completions.create(
        response_model=UserDetails,
        messages=[
            {"role": "user", "content": sample_data},
        ],
    )

    # Assertions
    assert response.name == expected_name, (
        f"Expected name {expected_name}, got {response.name}"
    )
    assert response.age == expected_age, (
        f"Expected age {expected_age}, got {response.age}"
    )
