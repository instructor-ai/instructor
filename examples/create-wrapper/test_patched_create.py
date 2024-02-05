import asyncio
import pytest_asyncio
import pytest
from itertools import product
from pydantic import BaseModel
from instructor.function_calls import Mode
from tests.openai.util import models, modes

from patched_create import patched_create, patched_create_delay_response_model


class UserDetails(BaseModel):
    name: str
    age: int


# Lists for models, test data, and modes
test_data = [
    ("Jason is 10", "Jason", 10),
    ("Alice is 25", "Alice", 25),
    ("Bob is 35", "Bob", 35),
]

@pytest.mark.asyncio
@pytest.mark.parametrize("model, data, mode", product(models, test_data, modes))
async def test_user_details(model, data, mode):
    sample_data, expected_name, expected_age = data

    if (mode, model) in {
        (Mode.JSON, "gpt-3.5-turbo"),
        (Mode.JSON, "gpt-4"),
    }:
        pytest.skip(f"{mode} mode is not supported for {model}, skipping test")
    
    # Setting up the client with the instructor patch
    # and patching the create function
    create = patched_create(UserDetails, mode=mode)

    # Calling the extract function with the provided model, sample data, and mode
    response = await create(
        model=model,
        messages=[
            {"role": "user", "content": sample_data},
        ],
    )
    
    # Now you can assert that the parsed details match the expected values
    # Assertions
    assert (
        response.name == expected_name
    ), f"Expected name {expected_name}, got {response.name}"
    assert (
        response.age == expected_age
    ), f"Expected age {expected_age}, got {response.age}"



@pytest.mark.asyncio
@pytest.mark.parametrize("model, data, mode", product(models, test_data, modes))
async def test_patched_create_delay_response_model(model, data, mode):
    sample_data, expected_name, expected_age = data

    if (mode, model) in {
        (Mode.JSON, "gpt-3.5-turbo"),
        (Mode.JSON, "gpt-4"),
    }:
        pytest.skip(f"{mode} mode is not supported for {model}, skipping test")

    # Setting up the client with the instructor patch
    create = patched_create_delay_response_model(mode=mode)

    # Calling the extract function with the provided model, sample data, and mode
    response = await create(
        UserDetails,
        model=model,
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
