from itertools import product
from collections.abc import Iterable
from pydantic import BaseModel
import pytest
import instructor
from writerai import AsyncWriter, Writer
from instructor.dsl.partial import Partial

from .util import models, modes


class UserExtract(BaseModel):
    first_name: str
    age: int


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_iterable_model(model: str, mode: instructor.Mode):
    client = instructor.from_writer(client=Writer(), mode=mode)
    response = client.chat.completions.create(
        model=model,
        response_model=Iterable[UserExtract],
        max_retries=2,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )

    models = []
    for m in response:
        assert isinstance(m, UserExtract)
        models += [m]

    assert len(models) == 2


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_writer_partial_model(model: str, mode: instructor.Mode):
    client = instructor.from_writer(client=Writer(), mode=mode)
    response = client.chat.completions.create(
        model=model,
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence : {{ name }} is {{ age }} years old",
            },
        ],
        context={"name": "Jason", "age": 12},
    )
    final_model = None
    for m in response:
        assert isinstance(m, UserExtract)
        final_model = m

    assert final_model.age == 12
    assert final_model.first_name == "Jason"


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_writer_iterable_model_async(model: str, mode: instructor.Mode):
    client = instructor.from_writer(client=AsyncWriter(), mode=mode)
    response = await client.chat.completions.create(
        model=model,
        response_model=Iterable[UserExtract],
        max_retries=2,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )

    models = []
    for m in response:
        assert isinstance(m, UserExtract)
        models += [m]

    assert len(models) == 2


@pytest.mark.parametrize("model,mode", product(models, modes))
@pytest.mark.asyncio
async def test_writer_partial_model_async(model: str, mode: instructor.Mode):
    client = instructor.from_writer(client=AsyncWriter(), mode=mode)
    response = await client.chat.completions.create(
        model=model,
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )
    final_model = None
    async for m in response:
        assert isinstance(m, UserExtract)
        final_model = m

    assert final_model.age == 12
    assert final_model.first_name == "Jason"
