from itertools import product

import pytest
from pydantic import BaseModel

import instructor
from instructor.dsl.partial import Partial

from .util import models, modes


class UserExtract(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_partial_model(model, mode, client):
    client = instructor.from_genai(client, mode=mode)
    model = client.chat.completions.create(
        model=model,
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "{{ name }} is {{ age }} years old"},
        ],
        context={"name": "Jason", "age": 12},
    )
    final_model = None
    for m in model:  # type: ignore
        assert isinstance(m, UserExtract)
        final_model = m
    assert isinstance(final_model, UserExtract)
    assert final_model.age == 12
    assert final_model.name == "Jason"


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_partial_model_async(client, model, mode):
    client = instructor.from_genai(client, mode=mode, use_async=True)
    response_stream = client.chat.completions.create_partial(
        model=model,
        response_model=UserExtract,
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


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_iterable_model(model, mode, client):
    client = instructor.from_genai(client, mode=mode)
    model = client.chat.completions.create_iterable(
        model=model,
        response_model=UserExtract,
        max_retries=2,
        stream=True,
        messages=[
            {
                "role": "user",
                "content": "{{ name }} is {{ age }} years old, James is 24 years old and Jackie is 22 years old",
            },
        ],
        context={"name": "Jason", "age": 12},
    )
    iterations = 0
    for m in model:  # type: ignore
        assert isinstance(m, UserExtract)
        iterations += 1
    assert iterations == 3


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_iterable_model_async(model, mode, client):
    client = instructor.from_genai(client, mode=mode, use_async=True)
    model = client.chat.completions.create_iterable(
        model=model,
        response_model=UserExtract,
        max_retries=2,
        stream=True,
        messages=[
            {
                "role": "user",
                "content": "{{ name }} is {{ age }} years old, James is 24 years old and Jackie is 22 years old",
            },
        ],
        context={"name": "Jason", "age": 12},
    )
    iterations = 0
    async for m in model:  # type: ignore
        assert isinstance(m, UserExtract)
        iterations += 1
    assert iterations == 3
