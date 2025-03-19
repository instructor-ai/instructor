import pytest
from pydantic import BaseModel
import instructor
from .util import modes, models


class User(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model", models)
def test_mistral_structured_outputs_sync(client, model, mode):
    patched_client = instructor.from_mistral(client, mode=mode)

    # Test extracting structured data
    response = patched_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Ivan is 27. Jack is 28. Jeffrey is 29."}
        ],
        response_model=list[User],
    )

    # Validate response
    assert isinstance(response, list)
    assert len(response) == 3

    # Check individual users
    assert {user.name for user in response} == {"Ivan", "Jack", "Jeffrey"}
    assert {user.age for user in response} == {27, 28, 29}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model", models)
async def test_mistral_structured_outputs_async(aclient, model, mode):
    # Apply instructor patch with MISTRAL_STRUCTURED_OUTPUTS mode
    patched_client = instructor.from_mistral(aclient, mode=mode, use_async=True)

    # Test extracting structured data
    response = await patched_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Ivan is 27. Jack is 28. Jeffrey is 29."}
        ],
        response_model=list[User],
    )

    # Validate response
    assert isinstance(response, list)
    assert len(response) == 3

    # Check individual users
    assert {user.name for user in response} == {"Ivan", "Jack", "Jeffrey"}
    assert {user.age for user in response} == {27, 28, 29}


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model", models)
def test_mistral_single_user_sync(client, model, mode):
    # Apply instructor patch with the specified mode
    patched_client = instructor.from_mistral(client, mode=mode)

    # Test extracting a single User object
    response = patched_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Ivan is 27 years old."}],
        response_model=User,
    )

    # Validate response
    assert isinstance(response, User)
    assert response.name == "Ivan"
    assert response.age == 27


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model", models)
async def test_mistral_single_user_async(aclient, model, mode):
    # Apply instructor patch with the specified mode
    patched_client = instructor.from_mistral(aclient, mode=mode, use_async=True)

    # Test extracting a single User object asynchronously
    response = await patched_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Jack is 28 years old."}],
        response_model=User,
    )

    # Validate response
    assert isinstance(response, User)
    assert response.name == "Jack"
    assert response.age == 28
