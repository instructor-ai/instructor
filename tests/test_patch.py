import pytest
import instructor

from pydantic import BaseModel, ValidationError, BeforeValidator
from openai import OpenAI, AsyncOpenAI
from instructor import llm_validator
from typing_extensions import Annotated


client = instructor.patch(OpenAI())
aclient = instructor.apatch(AsyncOpenAI())


@pytest.mark.asyncio
async def test_async_runmodel():
    class UserExtract(BaseModel):
        name: str
        age: int

    model = await aclient.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name.lower() == "jason"
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from OpenAI"


def test_runmodel():
    class UserExtract(BaseModel):
        name: str
        age: int

    model = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name.lower() == "jason"
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from OpenAI"


def test_runmodel_validator():
    from pydantic import field_validator

    class UserExtract(BaseModel):
        name: str
        age: int

        @field_validator("name")
        @classmethod
        def validate_name(cls, v):
            if v.upper() != v:
                raise ValueError("Name should be uppercase")
            return v

    model = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name == "JASON"
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from OpenAI"

@pytest.mark.asyncio
async def test_async_runmodel_validator():
    aclient = instructor.apatch(AsyncOpenAI())
    from pydantic import field_validator

    class UserExtract(BaseModel):
        name: str
        age: int

        @field_validator("name")
        @classmethod
        def validate_name(cls, v):
            if v.upper() != v:
                raise ValueError("Name should be uppercase")
            return v

    model = await aclient.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name == "JASON"
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from OpenAI"


def test_runmodel_validator_error():


    class QuestionAnswerNoEvil(BaseModel):
        question: str
        answer: Annotated[
            str,
            BeforeValidator(llm_validator("don't say objectionable things", openai_client=client))
        ]

    with pytest.raises(ValidationError):
        QuestionAnswerNoEvil(
            question="What is the meaning of life?",
            answer="The meaning of life is to be evil and steal",
        )