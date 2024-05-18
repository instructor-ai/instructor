from itertools import product
from collections.abc import Iterable
from pydantic import BaseModel
import pytest
import instructor
from instructor.dsl.partial import Partial

from .util import models, modes


class UserExtract(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model, mode, stream", product(models, modes, [True, False]))
def test_iterable_model(model, mode, stream, client):
    client = instructor.from_anthropic(client, mode=mode)
    model = client.messages.create(
        model=model,
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


@pytest.mark.parametrize("model, mode, stream", product(models, modes, [True, False]))
@pytest.mark.asyncio
async def test_iterable_model_async(model, mode, stream, aclient):
    aclient = instructor.from_anthropic(aclient, mode=mode)
    model = await aclient.messages.create(
        model=model,
        response_model=Iterable[UserExtract],
        max_retries=2,
        stream=stream,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    if stream:
        async for m in model:
            assert isinstance(m, UserExtract)
    else:
        for m in model:
            assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_partial_model(model, mode, client):
    client = instructor.from_anthropic(client, mode=mode)
    model = client.messages.create(
        model=model,
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
async def test_partial_model_async(model, mode, aclient):
    aclient = instructor.from_anthropic(aclient, mode=mode)
    model = await aclient.messages.create(
        model=model,
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
