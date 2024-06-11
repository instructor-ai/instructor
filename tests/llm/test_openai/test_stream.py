from itertools import product
from collections.abc import Iterable
from pydantic import BaseModel, Field
import pytest
import instructor
from instructor.dsl.partial import Partial

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


@pytest.mark.parametrize("model, mode, stream", product(models, modes, [True, False]))
@pytest.mark.asyncio
async def test_iterable_model_async(model, mode, stream, aclient):
    aclient = instructor.patch(aclient, mode=mode)
    model = await aclient.chat.completions.create(
        model=model,
        response_model=Iterable[UserExtract],
        max_retries=2,
        stream=stream,
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


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_summary_extraction(model, mode, client):
    class Summary(BaseModel):
        summary: str = Field(description="A detailed summary")

    client = instructor.from_openai(client, mode=mode)
    extraction_stream = client.chat.completions.create_partial(
        model=model,
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    for extraction in extraction_stream:
        if previous_summary is not None:
            assert extraction.summary.startswith(previous_summary)
            updates += 1
        previous_summary = extraction.summary

    assert updates > 1


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_summary_extraction_async(model, mode, aclient):
    class Summary(BaseModel):
        summary: str = Field(description="A detailed summary")

    client = instructor.from_openai(aclient, mode=mode)
    extraction_stream = client.chat.completions.create_partial(
        model=model,
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    async for extraction in extraction_stream:
        if previous_summary is not None:
            assert extraction.summary.startswith(previous_summary)
            updates += 1
        previous_summary = extraction.summary

    assert updates > 1
