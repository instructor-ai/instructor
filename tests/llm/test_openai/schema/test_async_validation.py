from pydantic import BaseModel, ValidationInfo
from instructor import async_field_validator, AsyncInstructMixin, async_model_validator
import pytest
import asyncio


@pytest.mark.asyncio
async def test_field_validator():
    class User(BaseModel, AsyncInstructMixin):
        name: str
        label: str

        @async_field_validator("name", "label")
        async def validate_user(self, v: str):
            await asyncio.sleep(3)
            if not v.isupper():
                raise ValueError(f"Uppercase response required for {v}")

    exceptions = await User(**{"name": "tom", "label": "active"}).model_async_validate()

    assert len(exceptions) == 2
    assert [str(item) for item in exceptions] == [
        "Exception of Uppercase response required for tom encountered at name",
        "Exception of Uppercase response required for active encountered at label",
    ]


@pytest.mark.asyncio
async def test_union_field_validator():
    class User(BaseModel, AsyncInstructMixin):
        name: str
        label: str

        @async_field_validator("name", "label")
        def validate_user(self, v: str):
            if not v.isupper():
                raise ValueError(f"Uppercase response required for {v}")

    exceptions = await User(**{"name": "tom", "label": "active"}).model_async_validate()

    assert len(exceptions) == 2
    assert [str(item) for item in exceptions] == [
        "Exception of Uppercase response required for tom encountered at name",
        "Exception of Uppercase response required for active encountered at label",
    ]


@pytest.mark.asyncio
async def test_model_validator():
    class User(BaseModel, AsyncInstructMixin):
        name: str
        label: str

        @async_model_validator()
        def validate_user(self):
            if not self.name.isupper() or not self.label.isupper():
                raise ValueError(f"Uppercase response required")

    exceptions = await User(**{"name": "tom", "label": "active"}).model_async_validate()

    assert len(exceptions) == 1
    assert [str(item) for item in exceptions] == [
        "Exception of Uppercase response required encountered",
    ]


@pytest.mark.asyncio
async def test_parsing_nested_field():
    class UserExtractValidated(BaseModel, AsyncInstructMixin):
        name: str
        age: int

        @async_field_validator("name")
        async def validate_name(cls, v: str) -> str:
            await asyncio.sleep(3)
            if not v.isupper():
                raise ValueError(
                    f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
                )
            return v

    class Users(BaseModel, AsyncInstructMixin):
        users: list[UserExtractValidated]

    exceptions = await Users(
        **{"users": [{"name": "thomas", "age": 27}, {"name": "vincent", "age": 24}]}
    ).model_async_validate()

    assert len(exceptions) == 2
    assert [str(item) for item in exceptions] == [
        "Exception of All Letters in the name must be uppercased. thomas is not a valid response. Eg JASON, TOM not jason, tom encountered at users.name",
        "Exception of All Letters in the name must be uppercased. vincent is not a valid response. Eg JASON, TOM not jason, tom encountered at users.name",
    ]


@pytest.mark.asyncio
async def test_context_passing_in_nested_model():
    class ModelValidationCheck(BaseModel, AsyncInstructMixin):
        user_names: list[str]

        @async_model_validator()
        async def validate_model(self, info: ValidationInfo):
            await asyncio.sleep(3)
            raise ValueError(f"Invalid Error but with {info.context}!")

    class ModelValidationWrapper(BaseModel, AsyncInstructMixin):
        model: ModelValidationCheck

    res = await ModelValidationWrapper(
        **{"model": {"user_names": ["Jack", "Thomas", "Ben"]}}
    ).model_async_validate(validation_context={"abcdef": "123"})

    assert len(res) == 1
    assert (
        str(res[0])
        == "Exception of Invalid Error but with {'abcdef': '123'}! encountered at model"
    )


@pytest.mark.asyncio
async def test_context_passing_in_nested_field_validator():
    class ModelValidationCheck(BaseModel, AsyncInstructMixin):
        user_names: list[str]

        @async_field_validator("user_names")
        async def validate_model(self, v: list[str], info: ValidationInfo):
            await asyncio.sleep(3)
            assert len(v) > 0
            raise ValueError(f"Invalid Error but with {info.context}!")

    class ModelValidationWrapper(BaseModel, AsyncInstructMixin):
        model: ModelValidationCheck

    res = await ModelValidationWrapper(
        **{"model": {"user_names": ["Jack", "Thomas", "Ben"]}}
    ).model_async_validate(validation_context={"abcdef": "123"})

    assert len(res) == 1
    assert (
        str(res[0])
        == "Exception of Invalid Error but with {'abcdef': '123'}! encountered at model.user_names"
    )
