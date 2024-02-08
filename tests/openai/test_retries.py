from typing import Annotated
from pydantic import AfterValidator, BaseModel, Field
import pytest
import instructor
from itertools import product
from tests.openai.util import models, modes


def uppercase_validator(v):
    if v.islower():
        raise ValueError("Name must be ALL CAPS")
    return v


class UserDetail(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)] = Field(
        ..., description="The name of the user"
    )
    age: int


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_upper_case_async(model, mode, aclient):
    client = instructor.patch(aclient, mode=mode)
    response = await client.chat.completions.create(
        model=model,
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=3,
    )
    assert response.name == "JASON"


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_upper_case_tenacity_async(model, mode, aclient):
    client = instructor.patch(aclient, mode=mode)
    from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed

    retries = AsyncRetrying(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
    )

    response = await client.chat.completions.create(
        model=model,
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=retries,
    )
    assert response.name == "JASON"


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_upper_case(model, mode, client):
    client = instructor.patch(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=3,
    )
    assert response.name == "JASON"


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_upper_case_tenacity(model, mode, client):
    client = instructor.patch(client, mode=mode)
    from tenacity import Retrying, stop_after_attempt, wait_fixed

    retries = Retrying(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
    )

    response = client.chat.completions.create(
        model=model,
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=retries,
    )
    assert response.name == "JASON"
