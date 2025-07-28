from itertools import product

import pytest
from pydantic import BaseModel

import instructor

from .util import models, modes


class UserExtract(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_partial_model(model, mode, client):
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)
    model = client.chat.completions.create_partial(
        model=model,
        response_model=UserExtract,
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
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=True)
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
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)
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
async def test_iterable_model_async(model, mode):
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=True)
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


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_iterable_model_with_thinking_config(model, mode, client):
    """Test that create_iterable works with thinking_config without raising unexpected keyword argument exception."""
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)

    thinking_config = {"thinkingBudget": 0}
    model_iter = client.chat.completions.create_iterable(
        model=model,
        response_model=UserExtract,
        max_retries=2,
        stream=True,
        thinking_config=thinking_config,
        messages=[
            {
                "role": "user",
                "content": "Jason is 25 years old, Alice is 30 years old",
            },
        ],
    )

    iterations = 0
    for m in model_iter:
        assert isinstance(m, UserExtract)
        iterations += 1
    assert iterations >= 1


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_iterable_model_with_thinking_config_async(model, mode):
    """Test that async create_iterable works with thinking_config without raising unexpected keyword argument exception."""
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=True)

    thinking_config = {"thinkingBudget": 0}
    model_iter = client.chat.completions.create_iterable(
        model=model,
        response_model=UserExtract,
        max_retries=2,
        stream=True,
        thinking_config=thinking_config,
        messages=[
            {
                "role": "user",
                "content": "Jason is 25 years old, Alice is 30 years old",
            },
        ],
    )

    iterations = 0
    async for m in model_iter:
        assert isinstance(m, UserExtract)
        iterations += 1
    assert iterations >= 1


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_create_with_thinking_config(model, mode, client):
    """Test that regular create works with thinking_config."""
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)

    thinking_config = {"thinkingBudget": 0}
    user = client.chat.completions.create(
        model=model,
        response_model=UserExtract,
        max_retries=2,
        thinking_config=thinking_config,
        messages=[
            {
                "role": "user",
                "content": "Jason is 25 years old",
            },
        ],
    )

    assert isinstance(user, UserExtract)
    assert user.name == "Jason"
    assert user.age == 25


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_create_with_thinking_config_async(model, mode):
    """Test that async create works with thinking_config."""
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=True)

    thinking_config = {"thinkingBudget": 0}
    user = await client.chat.completions.create(
        model=model,
        response_model=UserExtract,
        max_retries=2,
        thinking_config=thinking_config,
        messages=[
            {
                "role": "user",
                "content": "Jason is 25 years old",
            },
        ],
    )

    assert isinstance(user, UserExtract)
    assert user.name == "Jason"
    assert user.age == 25
