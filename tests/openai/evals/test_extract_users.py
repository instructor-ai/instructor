import pytest
from itertools import product
from pydantic import BaseModel
import instructor
from instructor.function_calls import Mode


class UserDetails(BaseModel):
    name: str
    age: int


# Lists for models, test data, and modes
models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]
test_data = [
    ("Jason is 10", "Jason", 10),
    ("Alice is 25", "Alice", 25),
    ("Bob is 35", "Bob", 35),
]
modes = [Mode.FUNCTIONS, Mode.JSON, Mode.TOOLS]


@pytest.mark.parametrize("model, data, mode", product(models, test_data, modes))
def test_extract(model, data, mode, client):
    sample_data, expected_name, expected_age = data

    if (mode, model) in {
        (Mode.JSON, "gpt-3.5-turbo"),
        (Mode.JSON, "gpt-4"),
    }:
        pytest.skip(f"{mode} mode is not supported for {model}, skipping test")

    # Setting up the client with the instructor patch
    client = instructor.patch(client, mode=mode)

    # Calling the extract function with the provided model, sample data, and mode
    response = client.chat.completions.create(
        model=model,
        response_model=UserDetails,
        messages=[
            {"role": "user", "content": sample_data},
        ],
    )

    # Assertions
    assert (
        response.name == expected_name
    ), f"Expected name {expected_name}, got {response.name}"
    assert (
        response.age == expected_age
    ), f"Expected age {expected_age}, got {response.age}"
