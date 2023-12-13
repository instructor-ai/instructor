from pydantic import BaseModel, field_validator
import pytest
import instructor

from openai import OpenAI, AsyncOpenAI

from instructor.function_calls import Mode

aclient = instructor.patch(AsyncOpenAI())
client = instructor.patch(OpenAI())


class UserExtract(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("mode", [Mode.FUNCTIONS, Mode.JSON, Mode.TOOLS, Mode.MD_JSON])
def test_runmodel(mode):
    client = instructor.patch(OpenAI(), mode=mode)
    model = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name.lower() == "jason"
    assert model.age == 25
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from OpenAI"


@pytest.mark.parametrize("mode", [Mode.FUNCTIONS, Mode.JSON, Mode.TOOLS, Mode.MD_JSON])
@pytest.mark.asyncio
async def test_runmodel_async(mode):
    aclient = instructor.patch(AsyncOpenAI(), mode=mode)
    model = await aclient.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name.lower() == "jason"
    assert model.age == 25
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from OpenAI"


class UserExtractValidated(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v.upper() != v:
            raise ValueError("Name should be uppercase")
        return v


@pytest.mark.parametrize("mode", [Mode.FUNCTIONS, Mode.JSON, Mode.MD_JSON])
def test_runmodel_validator(mode):
    client = instructor.patch(OpenAI(), mode=mode)
    model = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_model=UserExtractValidated,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtractValidated), "Should be instance of UserExtract"
    assert model.name == "JASON"
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from OpenAI"


@pytest.mark.parametrize("mode", [Mode.FUNCTIONS, Mode.JSON, Mode.MD_JSON])
@pytest.mark.asyncio
async def test_runmodel_async_validator(mode):
    aclient = instructor.patch(AsyncOpenAI(), mode=mode)
    model = await aclient.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_model=UserExtractValidated,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtractValidated), "Should be instance of UserExtract"
    assert model.name == "JASON"
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from OpenAI"
