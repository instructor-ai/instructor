from itertools import product
from pydantic import ValidationInfo
import pytest
import instructor
from openai import AsyncOpenAI
from instructor import from_openai
from ..util import models, modes
from instructor import async_field_validator, async_model_validator
from instructor.function_calls import openai_schema
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Union, Literal


class ValidationResult(BaseModel):
    chain_of_thought: str
    is_valid: bool


@pytest.mark.asyncio
@pytest.mark.skip
async def test_openai_schema_parser():
    class AdminUser(BaseModel):
        name: str
        age: int
        email: str

        @async_field_validator("name")
        def validate_name_admin(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
                )
            return v

    class User(BaseModel):
        name: str
        age: int

        @async_field_validator("name")
        def validate_name_user(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
                )
            return v

    class Users(BaseModel):
        users: list[Union[User, AdminUser]]

    resp = openai_schema(Users)(
        **{
            "users": [
                {
                    "name": "thomas",
                    "age": 27,
                },
                {"name": "vincent", "age": 24, "email": "vincent@gmail.com"},
            ]
        }
    )
    coros = await resp.get_model_coroutines()  # type: ignore
    # We should extract out two separate coros to show that we've handled a union type well
    assert len(coros) == 2


class User(BaseModel):
    name: str = Field(description="User's name")
    age: int = Field(description="User's age")
    email: str = Field(description="User's email address")


@pytest.mark.skip
def test_openai_schema_serialization():
    UserSchema = openai_schema(User)
    assert User.model_json_schema() == UserSchema.model_json_schema()


@pytest.mark.skip
def test_openai_schema_float_and_bool():
    class FloatBoolModel(BaseModel):
        price: float = Field(description="Price of an item")
        is_available: bool = Field(description="Availability status")

    FloatBoolSchema = openai_schema(FloatBoolModel)
    assert FloatBoolModel.model_json_schema() == FloatBoolSchema.model_json_schema()


@pytest.mark.skip
def test_openai_schema_bytes_and_literal():
    class BytesLiteralModel(BaseModel):
        data: bytes = Field(description="Binary data")
        status: Literal["active", "inactive"] = Field(description="Current status")

    BytesLiteralSchema = openai_schema(BytesLiteralModel)
    assert (
        BytesLiteralModel.model_json_schema() == BytesLiteralSchema.model_json_schema()
    )


@pytest.mark.skip
def test_nested_class():
    class Users(BaseModel):
        users: list[User]
        user: User

    assert Users.model_json_schema() == openai_schema(Users).model_json_schema()


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
def test_has_async_validators():
    class UserWithAsyncValidators(BaseModel):
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

    class UserWithoutAsyncValidators(BaseModel):
        name: str
        age: int

    user_with = openai_schema(UserWithAsyncValidators)(name="John", age=30)
    user_without = openai_schema(UserWithoutAsyncValidators)(name="Jane", age=25)

    assert user_with.has_async_validators() == True
    assert user_without.has_async_validators() == False

    class NestedUsers(BaseModel):
        user_with: UserWithAsyncValidators
        user_without: UserWithoutAsyncValidators

    nested_users = openai_schema(NestedUsers)(
        **{
            "user_with": {"name": "John", "age": 30},
            "user_without": {"name": "Jane", "age": 25},
        }
    )

    assert nested_users.has_async_validators() == True

    class AllWithoutValidators(BaseModel):
        users: list[UserWithoutAsyncValidators]

    all_without = openai_schema(AllWithoutValidators)(
        **{
            "users": [
                {"name": "Jane", "age": 25},
                {"name": "Jane", "age": 25},
            ]
        }
    )
    assert all_without.has_async_validators() == False


@pytest.mark.skip
def test_schema_optional_and_enum():
    class QueryType(str, Enum):
        DOCUMENT_CONTENT = "document_content"
        LAST_MODIFIED = "last_modified"
        ACCESS_PERMISSIONS = "access_permissions"
        RELATED_DOCUMENTS = "related_documents"

    # Define the structure for query responses
    class QueryResponse(BaseModel):
        query_type: QueryType
        response: str
        additional_info: Optional[str] = None

    assert (
        openai_schema(QueryResponse).model_json_schema()
        == QueryResponse.model_json_schema()
    )


@pytest.mark.skip
def test_schema_union():
    class Search(BaseModel):
        search_query: str

    class CalendarInvite(BaseModel):
        date: str

    # Define the structure for query responses
    class QueryResponse(BaseModel):
        query_type: Union[Search, CalendarInvite]
        response: str
        additional_info: Optional[str] = None

    assert (
        openai_schema(QueryResponse).model_json_schema()
        == QueryResponse.model_json_schema()
    )
