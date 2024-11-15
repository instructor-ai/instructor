from itertools import product
from pydantic import BaseModel, field_validator
from typing_extensions import TypedDict
from writerai import AsyncWriter, Writer
import pytest
import instructor

from .util import models, modes


class UserExtract(BaseModel):
    first_name: str
    age: int


class UserExtractTypedDict(TypedDict):
    first_name: str
    age: int


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_typed_dict(model: str, mode: instructor.Mode):
    create_fn = Writer().chat.chat
    new_create_fn = instructor.patch(create=create_fn, mode=mode)

    response = new_create_fn(
        model=model,
        response_model=UserExtractTypedDict,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract Jason is 25 years old"},
        ],
    )
    assert isinstance(response, BaseModel), "Should be instance of a pydantic model"
    assert response.first_name == "Jason"
    assert response.age == 25


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_runmodel(model: str, mode: instructor.Mode):
    create_fn = Writer().chat.chat
    new_create_fn = instructor.patch(create=create_fn, mode=mode)

    response = new_create_fn(
        model=model,
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract Jason is 25 years old"},
        ],
    )
    assert isinstance(response, UserExtract), "Should be instance of UserExtract"
    assert response.first_name == "Jason"
    assert response.age == 25


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_writer_runmodel_async(model: str, mode: instructor.Mode):
    create_fn = AsyncWriter().chat.chat
    new_create_fn = instructor.patch(create=create_fn, mode=mode)

    response = await new_create_fn(
        model=model,
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract Jason is 25 years old"},
        ],
    )
    assert isinstance(response, UserExtract), "Should be instance of UserExtract"
    assert response.first_name == "Jason"
    assert response.age == 25



class UserExtractValidated(BaseModel):
    first_name: str
    age: int

    @field_validator("first_name")
    @classmethod
    def validate_name(cls, v):
        if v.upper() != v:
            raise ValueError(
                "Name should have all letters in uppercase. Make sure to use the `uppercase` form of the name"
            )
        return v


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_runmodel_validator(model: str, mode: instructor.Mode):
    create_fn = Writer().chat.chat
    new_create_fn = instructor.patch(create=create_fn, mode=mode)

    response = new_create_fn(
        model=model,
        response_model=UserExtractValidated,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(response, UserExtractValidated), "Should be instance of UserExtract"
    assert response.first_name == "JASON"


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_writer_runmodel_async_validator(model: str, mode: instructor.Mode):
    create_fn = AsyncWriter().chat.chat
    new_create_fn = instructor.patch(create=create_fn, mode=mode)

    response = await new_create_fn(
        model=model,
        response_model=UserExtractValidated,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(response, UserExtractValidated), "Should be instance of UserExtract"
    assert response.first_name == "JASON"
