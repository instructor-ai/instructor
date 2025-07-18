from itertools import product
from collections.abc import Iterable
from pydantic import BaseModel
import pytest
import instructor
from instructor.dsl.partial import Partial
from typing import Union, Literal
from .util import models, modes


class UserExtract(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model, mode, stream", product(models, modes, [True, False]))
def test_iterable_model(model, mode, stream):
    client = instructor.from_provider(model, mode=mode)
    model = client.create(
        response_model=Iterable[UserExtract],
        max_retries=2,
        stream=stream,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    for m in model:
        assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_iterable_model_async(model, mode):
    client = instructor.from_provider(model, mode=mode, async_client=True)
    model = await client.chat.completions.create(
        response_model=Iterable[UserExtract],
        max_retries=2,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    async for m in model:
        assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_partial_model(model, mode):
    client = instructor.from_provider(model, mode=mode)
    model = client.create(
        response_model=Partial[UserExtract],
        max_retries=2,
        max_tokens=1024,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )
    for m in model:
        assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model,mode", product(models, modes))
@pytest.mark.asyncio
async def test_partial_model_async(model, mode):
    client = instructor.from_provider(model, mode=mode, async_client=True)
    model = await client.create(
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )
    async for m in model:
        assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_model(model, mode):
    client = instructor.from_provider(model, mode=mode)
    model = client.create(
        response_model=UserExtract,
        max_retries=2,
        max_tokens=1024,
        system="You are a helpful assistant that extracts user information from sentences",
        messages=[{"role": "user", "content": "Jason is 8 years old"}],
    )

    assert model.name == "Jason"
    assert model.age == 8


@pytest.mark.parametrize("model,mode", product(models, modes))
@pytest.mark.asyncio
async def test_model_async(model, mode):
    client = instructor.from_provider(model, mode=mode, async_client=True)
    model = await client.create(
        response_model=UserExtract,
        max_retries=2,
        max_tokens=1024,
        system="You are a helpful assistant that extracts user information from sentences",
        messages=[{"role": "user", "content": "Jason is 8 years old"}],
    )

    assert model.name == "Jason"
    assert model.age == 8


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_sync_iterable_union_model(model, mode):
    client = instructor.from_provider(model, mode=mode)
    model = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "What is the weather in toronto, dallas and what is the tallest building in the world?",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
        max_tokens=1024,
    )
    count = 0
    for m in model:
        assert isinstance(m, (Weather, GoogleSearch))
        count += 1
    assert count == 3


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_iterable_create_union_model(model, mode):
    client = instructor.from_provider(model, mode=mode)
    model = client.chat.completions.create_iterable(
        messages=[
            {
                "role": "system",
                "content": "You must always use tools and use a single weather call for each city",
            },
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Union[Weather, GoogleSearch],
        max_tokens=1024,
    )
    count = 0
    for m in model:
        assert isinstance(m, (Weather, GoogleSearch))
        count += 1
    assert count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("model, mode", product(models, modes))
async def test_async_iterable_union_model(model, mode):
    client = instructor.from_provider(model, mode=mode, async_client=True)
    model = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You must always use tools and use a single weather call for each city",
            },
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
        max_tokens=1024,
    )
    count = 0
    async for m in model:
        assert isinstance(m, (Weather, GoogleSearch))
        count += 1
    assert count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("model, mode", product(models, modes))
async def test_async_iterable_create_union_model(model, mode):
    client = instructor.from_provider(model, mode=mode, async_client=True)
    model = client.chat.completions.create_iterable(
        messages=[
            {
                "role": "system",
                "content": "You must always use tools and use a single weather call for each city",
            },
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Union[Weather, GoogleSearch],
        max_tokens=1024,
    )
    count = 0
    async for m in model:
        assert isinstance(m, (Weather, GoogleSearch))
        count += 1
    assert count == 3
