from typing import Annotated
import pytest
from pydantic import AfterValidator, BaseModel, Field
import instructor
from .util import models, modes
from itertools import product


def uppercase_validator(v):
    if v.islower():
        raise ValueError("Name must be ALL CAPS")
    return v


class UserDetail(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)] = Field(
        ..., description="The name of the user"
    )
    age: int


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
@pytest.mark.asyncio
async def test_upper_case(aclient, model, mode):
    aclient = instructor.from_genai(aclient, mode=mode, use_async=True)
    response = await aclient.chat.completions.create(
        response_model=UserDetail,
        model=model,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=3,
    )
    assert response.name == "JASON"


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
@pytest.mark.asyncio
async def test_upper_case_tenacity(aclient, model, mode):
    aclient = instructor.from_genai(aclient, mode=mode, use_async=True)
    from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed

    retries = AsyncRetrying(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
    )

    response = await aclient.chat.completions.create(
        response_model=UserDetail,
        model=model,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=retries,
    )
    assert response.name == "JASON"


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_upper_case_sync(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        response_model=UserDetail,
        model=model,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=3,
    )
    assert response.name == "JASON"


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_upper_case_tenacity_sync(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    from tenacity import Retrying, stop_after_attempt, wait_fixed

    retries = Retrying(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
    )

    response = client.chat.completions.create(
        response_model=UserDetail,
        model=model,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=retries,
    )
    assert response.name == "JASON"
