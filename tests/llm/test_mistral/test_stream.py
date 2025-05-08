from itertools import product
from pydantic import BaseModel
import pytest
import instructor
from instructor.dsl.partial import Partial

from .util import models, modes


class UserExtract(BaseModel):
    name: str
    age: int


from typing import Union, Literal
from collections.abc import Iterable


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_mistral_iterable_model(model, mode, client):
    client = instructor.from_mistral(client, mode=mode)
    model = client.chat.completions.create_iterable(
        model=model,
        response_model=UserExtract,
        max_retries=1,
        messages=[
            {"role": "user", "content": "Make two people up"},
        ],
    )
    iterations = 0
    for m in model:
        assert isinstance(m, UserExtract)
        iterations += 1
    assert iterations == 2


@pytest.mark.parametrize(
    "model, mode", product(models, [instructor.Mode.MISTRAL_STRUCTURED_OUTPUTS])
)
def test_mistral_iterable_union_model(model, mode, client):
    client = instructor.from_mistral(client, mode=mode)
    model = client.chat.completions.create_iterable(
        model=model,
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Union[Weather, GoogleSearch],
    )
    for m in model:
        assert isinstance(m, (Weather, GoogleSearch))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, mode", product(models, [instructor.Mode.MISTRAL_STRUCTURED_OUTPUTS])
)
async def test_mistral_async_iterable_union_model(model, mode, aclient):
    client = instructor.from_mistral(aclient, mode=mode, use_async=True)
    model = client.chat.completions.create_iterable(
        model=model,
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Union[Weather, GoogleSearch],
    )
    async for m in model:
        assert isinstance(m, (Weather, GoogleSearch))


@pytest.mark.parametrize(
    "model, mode", product(models, [instructor.Mode.MISTRAL_STRUCTURED_OUTPUTS])
)
def test_mistral_sync_iterable_union_model(model, mode, client):
    client = instructor.from_mistral(client, mode=mode)
    model = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
    )
    for m in model:
        assert isinstance(m, (Weather, GoogleSearch))


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_mistral_iterable_model_async(model, mode, aclient):
    aclient = instructor.from_mistral(aclient, mode=mode, use_async=True)
    model = aclient.chat.completions.create_iterable(
        model=model,
        response_model=UserExtract,
        max_retries=1,
        messages=[
            {"role": "user", "content": "Make two people up"},
        ],
    )
    iterations = 0
    async for m in model:
        assert isinstance(m, UserExtract)
        iterations += 1
    assert iterations == 2


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_mistral_partial_model(model, mode, client):
    client = instructor.from_mistral(client, mode=mode)
    model = client.chat.completions.create(
        model=model,
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )
    iterations = 0
    for m in model:
        iterations += 1
        assert isinstance(m, UserExtract)
    assert iterations >= 1


@pytest.mark.parametrize("model,mode", product(models, modes))
@pytest.mark.asyncio
async def test_mistral_partial_model_async(model, mode, aclient):
    aclient = instructor.from_mistral(aclient, mode=mode, use_async=True)
    model = await aclient.chat.completions.create(
        model=model,
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )
    iterations = 0
    async for m in model:
        iterations += 1
        assert isinstance(m, UserExtract)
    assert iterations >= 1
