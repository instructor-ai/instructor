from collections.abc import Iterable
from itertools import product
from pydantic import BaseModel
import pytest
import instructor
import vertexai.generative_models as gm  # type: ignore
from instructor.dsl.partial import Partial

from .util import models, modes


class UserExtract(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_iterable_model(model, mode):
    client = instructor.from_vertexai(gm.GenerativeModel(model), mode)
    response_stream = client.chat.completions.create(
        response_model=Iterable[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    iterations = 0
    for chunk in response_stream:
        assert isinstance(chunk, UserExtract)
        iterations += 1
    assert iterations == 2


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_iterable_model_async(model, mode):
    client = instructor.from_vertexai(gm.GenerativeModel(model), mode, _async=True)
    response_stream = await client.chat.completions.create_iterable(
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    iterations = 0
    async for chunk in response_stream:
        assert isinstance(chunk, UserExtract)
        iterations += 1
    assert iterations == 2


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_partial_model(model, mode):
    client = instructor.from_vertexai(gm.GenerativeModel(model), mode)
    response_stream = client.chat.completions.create(
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Anibal is 23 years old"},
        ],
    )
    iterations = 0
    for chunk in response_stream:
        assert isinstance(chunk, UserExtract)
        iterations += 1
    assert iterations >= 1


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_partial_model_async(model, mode):
    client = instructor.from_vertexai(gm.GenerativeModel(model), mode, _async=True)
    response_stream = await client.chat.completions.create(
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Anibal is 23 years old"},
        ],
    )
    iterations = 0
    async for chunk in response_stream:
        assert isinstance(chunk, UserExtract)
        iterations += 1
    assert iterations >= 1
