import pytest
from pydantic import BaseModel, field_validator
import instructor
from .util import modes, models


class User(BaseModel):
    name: str
    age: int

    @field_validator("age")
    def validate_age(cls, v):
        if v > 0:
            raise ValueError(
                "Age must be expressed as a negative number (Eg. 25 is -25 )"
            )
        return v


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model", models)
def test_mistral_retry_validation(client, model, mode):
    patched_client = instructor.from_mistral(client, mode=mode)

    # Test extracting structured data with validation that should trigger retry
    response = patched_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Ivan is 25 years old"}],
        response_model=User,
    )

    # Validate response has correct negative age after retry
    assert isinstance(response, User)
    assert response.name == "Ivan"
    assert response.age == -25


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model", models)
async def test_mistral_retry_validation_async(aclient, model, mode):
    patched_client = instructor.from_mistral(aclient, mode=mode, use_async=True)

    # Test extracting structured data with validation that should trigger retry
    response = await patched_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Jack is 28 years old"}],
        response_model=User,
    )

    # Validate response has correct negative age after retry
    assert isinstance(response, User)
    assert response.name == "Jack"
    assert response.age == -28
