from itertools import product
from pydantic import ValidationInfo
import pytest
import instructor
from openai import AsyncOpenAI
from instructor import from_openai
from .util import models, modes
from instructor import async_field_validator, async_model_validator
from instructor.function_calls import openai_schema
from instructor.function_calls import OpenAISchema
from pydantic import BaseModel, Field


class UserExtractValidated(OpenAISchema):
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


class ValidationResult(OpenAISchema):
    chain_of_thought: str
    is_valid: bool


class ExtractedContent(OpenAISchema):
    relevant_question: str

    @async_field_validator("relevant_question")
    async def validate_relevant_question(cls, v: str, info: ValidationInfo) -> str:
        client = from_openai(AsyncOpenAI())
        if info.context and "content" in info.context:
            original_source = info.context["content"]
            assert (
                len(original_source) > 10
            )  # Asserting that a valid string was indeed passed in
            evaluation = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Evaluate and determine if the question is a valid question well supported from the text",
                    },
                    {
                        "role": "user",
                        "content": f"The question is {v} and the source is {original_source}",
                    },
                ],
                response_model=ValidationResult,
            )
            if not evaluation.is_valid:
                raise ValueError(f"{v} is an invalid question!")
            return v

        raise ValueError("Invalid Response!")


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_async_validator(model, mode, aclient):
    aclient = instructor.from_openai(aclient, mode=mode)
    content = """
    From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. 

    Groups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland. 

    In this struggle as President Zelenskyy said in his speech to the European Parliament “Light will win over darkness.” The Ukrainian Ambassador to the United States is here tonight. 

    Let each of us here tonight in this Chamber send an unmistakable signal to Ukraine and to the world. 

    Please rise if you are able and show that, Yes, we the United States of America stand with the Ukrainian people. 

    Throughout our history we’ve learned this lesson when dictators do not pay a price for their aggression they cause more chaos.   

    They keep moving.
    """
    model = await aclient.chat.completions.create(
        model=model,
        response_model=ExtractedContent,
        max_retries=2,
        messages=[
            {
                "role": "user",
                "content": f"Generate a question from the context of {content}",
            },
        ],
        validation_context={"content": content},
    )
    assert isinstance(
        model, ExtractedContent
    ), "Should be instance of Extracted Content"


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_nested_model(model, mode, aclient):
    class Users(OpenAISchema):
        users: list[UserExtractValidated]

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


@pytest.mark.asyncio
async def test_field_validator():
    class User(OpenAISchema):
        name: str
        label: str

        @async_field_validator("name", "label")
        def validate_user(self, v: str):
            if not v.isupper():
                raise ValueError(f"Uppercase response required for {v}")

    exceptions = await User(name="tom", label="active").model_async_validate()

    assert len(exceptions) == 2
    assert [str(item) for item in exceptions] == [
        "Exception of Uppercase response required for tom encountered at name",
        "Exception of Uppercase response required for active encountered at label",
    ]


@pytest.mark.asyncio
async def test_model_validator():
    class User(OpenAISchema):
        name: str
        label: str

        @async_model_validator()
        def validate_user(self):
            if not self.name.isupper() or not self.label.isupper():
                raise ValueError(f"Uppercase response required")

    exceptions = await User(name="tom", label="active").model_async_validate()

    assert len(exceptions) == 1
    assert [str(item) for item in exceptions] == [
        "Exception of Uppercase response required encountered",
    ]


@pytest.mark.asyncio
async def test_parsing_nested_field():
    class Users(OpenAISchema):
        users: list[UserExtractValidated]

    exceptions = await Users(
        users=[
            UserExtractValidated(name="thomas", age=22),
            UserExtractValidated(name="vincent", age=24),
        ]
    ).model_async_validate()

    assert len(exceptions) == 2
    assert [str(item) for item in exceptions] == [
        "Exception of All Letters in the name must be uppercased. thomas is not a valid response. Eg JASON, TOM not jason, tom encountered at users.name",
        "Exception of All Letters in the name must be uppercased. vincent is not a valid response. Eg JASON, TOM not jason, tom encountered at users.name",
    ]


@pytest.mark.asyncio
async def test_context_passing_in_nested_model():
    class ModelValidationCheck(OpenAISchema):
        user_names: list[str]

        @async_model_validator()
        def validate_model(self, info: ValidationInfo):
            raise ValueError(f"Invalid Error but with {info.context}!")

    class ModelValidationWrapper(OpenAISchema):
        model: ModelValidationCheck

    res = await ModelValidationWrapper(
        model=ModelValidationCheck(user_names=["Jack", "Thomas", "Ben"])
    ).model_async_validate(validation_context={"abcdef": "123"})

    assert len(res) == 1
    assert (
        str(res[0])
        == "Exception of Invalid Error but with {'abcdef': '123'}! encountered at model"
    )


@pytest.mark.asyncio
async def test_context_passing_in_nested_field_validator():
    class ModelValidationCheck(OpenAISchema):
        user_names: list[str]

        @async_field_validator("user_names")
        def validate_model(self, v: list[str], info: ValidationInfo):
            assert len(v) > 0
            raise ValueError(f"Invalid Error but with {info.context}!")

    class ModelValidationWrapper(OpenAISchema):
        model: ModelValidationCheck

    res = await ModelValidationWrapper(
        model=ModelValidationCheck(user_names=["Jack", "Thomas", "Ben"])
    ).model_async_validate(validation_context={"abcdef": "123"})

    assert len(res) == 1
    assert (
        str(res[0])
        == "Exception of Invalid Error but with {'abcdef': '123'}! encountered at model.user_names"
    )


from typing import Literal
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(description="User's name")
    age: int = Field(description="User's age")
    email: str = Field(description="User's email address")


def test_openai_schema_serialization():
    UserSchema = openai_schema(User)
    assert User.model_json_schema() == UserSchema.model_json_schema()


def test_openai_schema_float_and_bool():
    class FloatBoolModel(BaseModel):
        price: float = Field(description="Price of an item")
        is_available: bool = Field(description="Availability status")

    FloatBoolSchema = openai_schema(FloatBoolModel)
    assert FloatBoolModel.model_json_schema() == FloatBoolSchema.model_json_schema()


def test_openai_schema_bytes_and_literal():
    class BytesLiteralModel(BaseModel):
        data: bytes = Field(description="Binary data")
        status: Literal["active", "inactive"] = Field(description="Current status")

    BytesLiteralSchema = openai_schema(BytesLiteralModel)
    assert (
        BytesLiteralModel.model_json_schema() == BytesLiteralSchema.model_json_schema()
    )


def test_nested_class():
    class Users(BaseModel):
        users: list[User]
        user: User

    assert Users.model_json_schema() == openai_schema(Users).model_json_schema()


def test_nested_class_with_async_decorators():
    class NestedUserWithValidation(BaseModel):
        name: str

        @async_field_validator("name")
        async def validate_uppercase(self, v: str) -> str:
            await asyncio.sleep(2)
            return v

    class Users(BaseModel):
        users: list[NestedUserWithValidation]

    assert Users.model_json_schema() == openai_schema(Users).model_json_schema()


def test_nested_class_with_multiple_async_decorators():
    class User(BaseModel):
        name: str
        age: int

        @async_model_validator()
        async def validate_user(self) -> "User":
            await asyncio.sleep(1)
            if self.age < 0:
                raise ValueError("Age must be non-negative")
            return self

    class Users(BaseModel):
        users: list[User]
        user: User

        @async_model_validator()
        async def validate_users(self) -> "Users":
            await asyncio.sleep(1)
            if len(self.users) == 0:
                raise ValueError("Users list cannot be empty")
            return self

    assert Users.model_json_schema() == openai_schema(Users).model_json_schema()


def test_has_async_validators():
    class UserWithAsyncValidators(OpenAISchema):
        name: str
        age: int

        @async_field_validator("name")
        async def validate_name(cls, v: str) -> str:
            await asyncio.sleep(0.1)
            return v.strip()

        @async_model_validator()
        async def validate_age(self) -> "UserWithAsyncValidators":
            await asyncio.sleep(0.1)
            if self.age < 0:
                raise ValueError("Age must be non-negative")
            return self

    class UserWithoutAsyncValidators(OpenAISchema):
        name: str
        age: int

    user_with = UserWithAsyncValidators(name="John", age=30)
    user_without = UserWithoutAsyncValidators(name="Jane", age=25)

    assert user_with.has_async_validators() == True
    assert user_without.has_async_validators() == False

    class NestedUsers(OpenAISchema):
        user_with: UserWithAsyncValidators
        user_without: UserWithoutAsyncValidators

    nested_users = NestedUsers(user_with=user_with, user_without=user_without)
    assert nested_users.has_async_validators() == True

    class AllWithoutValidators(OpenAISchema):
        users: list[UserWithoutAsyncValidators]

    all_without = AllWithoutValidators(users=[user_without, user_without])
    assert all_without.has_async_validators() == False
