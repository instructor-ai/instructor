from pydantic import BaseModel, ValidationInfo
from instructor import async_field_validator, AsyncInstructMixin
from instructor.exceptions import InstructorRetryException
from ..util import models, modes
import pytest
import instructor
from itertools import product


class UserExtractValidated(BaseModel, AsyncInstructMixin):
    name: str
    age: int

    @async_field_validator("name")
    async def validate_name(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError(
                f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
            )
        return v


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_simple_validator(model, mode, aclient):
    aclient = instructor.from_openai(aclient, mode=mode)
    model = await aclient.chat.completions.create(
        model=model,
        response_model=UserExtractValidated,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtractValidated), "Should be instance of UserExtract"
    assert model.name == "JASON"


class ExtractedContent(BaseModel, AsyncInstructMixin):
    relevant_question: str

    @async_field_validator("relevant_question")
    async def validate_relevant_question(cls, v: str, info: ValidationInfo) -> str:
        if info.context and "content" in info.context:
            raise ValueError(f"Invalid context of {info.context['content']}")


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_async_validator_context(model, mode, aclient):
    aclient = instructor.from_openai(aclient, mode=mode)
    with pytest.raises(InstructorRetryException) as exc_info:
        model = await aclient.chat.completions.create(
            model=model,
            response_model=ExtractedContent,
            max_retries=1,
            messages=[
                {"role": "user", "content": "Extract jason is 25 years old"},
            ],
            validation_context={"content": "This is a test content"},
        )

    assert (
        exc_info.value.__cause__.__cause__.args[0]
        == "Validation errors: [ValueError('Exception of Invalid context of This is a test content encountered at relevant_question')]"
    )


class Users(BaseModel, AsyncInstructMixin):
    users: list[UserExtractValidated]


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_nested_model(model, mode, aclient):
    aclient = instructor.from_openai(aclient, mode=mode)
    resp = await aclient.chat.completions.create(
        model=model,
        response_model=Users,
        messages=[
            {
                "role": "user",
                "content": f"Extract users from this sentence - Tom is 22 and lives with his roomate Jack who is 24",
            }
        ],
    )

    assert isinstance(resp, Users)
    for user in resp.users:
        assert user.name.isupper()
