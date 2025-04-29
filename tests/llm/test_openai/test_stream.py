from itertools import product
from typing import Literal, Union
from collections.abc import Iterable
from pydantic import BaseModel
import pytest
import instructor
from instructor.dsl.partial import Partial, PartialLiteralMixin

from .util import models, modes


class UserExtract(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model, mode, stream", product(models, modes, [True, False]))
def test_iterable_model(model, mode, stream, client):
    client = instructor.patch(client, mode=mode)
    model = client.chat.completions.create(
        model=model,
        response_model=Iterable[UserExtract],
        max_retries=2,
        stream=stream,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    for m in model:
        assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_iterable_model_async(model, mode, aclient):
    aclient = instructor.from_openai(aclient, mode=mode)
    model = await aclient.chat.completions.create(
        model=model,
        response_model=Iterable[UserExtract],
        max_retries=2,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )

    async for m in model:
        assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_partial_model(model, mode, client):
    client = instructor.patch(client, mode=mode)
    model = client.chat.completions.create(
        model=model,
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )
    for m in model:
        assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model,mode", product(models, modes))
@pytest.mark.asyncio
async def test_partial_model_async(model, mode, aclient):
    aclient = instructor.patch(aclient, mode=mode)
    model = await aclient.chat.completions.create(
        model=model,
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )
    async for m in model:
        assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_literal_partial_mixin(model, mode, client):
    class UserWithMixin(BaseModel, PartialLiteralMixin):
        name: str
        age: int

    client = instructor.patch(client, mode=mode)
    resp = client.chat.completions.create(
        model=model,
        response_model=Partial[UserWithMixin],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )

    changes = 0
    last_name = None
    last_age = None
    for m in resp:
        assert isinstance(m, UserWithMixin)
        if m.name != last_name:
            last_name = m.name
            changes += 1
        if m.age != last_age:
            last_age = m.age
            changes += 1

    assert changes == 2  # Ensure we got at least one field update

    class UserWithoutMixin(BaseModel):
        name: str
        age: int

    resp = client.chat.completions.create(
        model=model,
        response_model=Partial[UserWithoutMixin],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )

    changes = 0
    last_name = None
    last_age = None
    for m in resp:
        assert isinstance(m, UserWithoutMixin)
        if m.name != last_name:
            last_name = m.name
            changes += 1
        if m.age != last_age:
            last_age = m.age
            changes += 1

    assert changes > 3


@pytest.mark.asyncio
@pytest.mark.parametrize("model,mode", product(models, modes))
async def test_literal_partial_mixin_async(model, mode, aclient):
    class UserWithMixin(BaseModel, PartialLiteralMixin):
        name: str
        age: int

    client = instructor.patch(aclient, mode=mode)
    resp = await client.chat.completions.create(
        model=model,
        response_model=Partial[UserWithMixin],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )

    changes = 0
    last_name = None
    last_age = None
    async for m in resp:
        assert isinstance(m, UserWithMixin)
        if m.name != last_name:
            last_name = m.name
            changes += 1
        if m.age != last_age:
            last_age = m.age
            changes += 1

    assert changes == 2  # Ensure we got at least one field update

    class UserWithoutMixin(BaseModel):
        name: str
        age: int

    resp = await client.chat.completions.create(
        model=model,
        response_model=Partial[UserWithoutMixin],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )

    changes = 0
    last_name = None
    last_age = None
    async for m in resp:
        assert isinstance(m, UserWithoutMixin)
        if m.name != last_name:
            last_name = m.name
            changes += 1
        if m.age != last_age:
            last_age = m.age
            changes += 1

    assert changes > 3


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


@pytest.mark.asyncio
@pytest.mark.parametrize("model, mode", product(models, modes))
async def test_async_iterable_union_model(model, mode, aclient):
    client = instructor.from_openai(aclient, mode=mode)
    model = await client.chat.completions.create(
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
    async for m in model:
        assert isinstance(m, (Weather, GoogleSearch))


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_sync_iterable_union_model(model, mode, client):
    client = instructor.from_openai(client, mode=mode)
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
def test_iterable_union_model(model, mode, client):
    client = instructor.from_openai(client, mode=mode)
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
@pytest.mark.parametrize("model, mode", product(models, modes))
async def test_async_iterable_create_union_model(model, mode, aclient):
    client = instructor.from_openai(aclient, mode=mode)
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
