from pydantic import BaseModel, Field
from instructor.function_calls import openai_schema
from instructor.decorators import async_field_validator, async_model_validator
import asyncio


class User(BaseModel):
    name: str = Field(description="User's name")
    age: int = Field(description="User's age")
    email: str = Field(description="User's email address")


def test_openai_schema_serialization():
    UserSchema = openai_schema(User)
    assert User.model_json_schema() == UserSchema.model_json_schema()


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
        async def validate_user(self):
            await asyncio.sleep(1)
            if self.age < 0:
                raise ValueError("Age must be non-negative")
            return self

    class Users(BaseModel):
        users: list[User]
        user: User

        @async_model_validator()
        async def validate_users(self):
            await asyncio.sleep(1)
            if len(self.users) == 0:
                raise ValueError("Users list cannot be empty")
            return self

    assert Users.model_json_schema() == openai_schema(Users).model_json_schema()
