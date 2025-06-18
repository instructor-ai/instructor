import pytest
from typing import Optional, Union
from pydantic import BaseModel
import instructor

from instructor.utils import map_to_gemini_function_schema


def test_optional_types_allowed():
    """Test that Optional types are correctly mapped and don't throw errors."""

    class User(BaseModel):
        name: str
        age: Optional[int] = None
        email: Optional[str] = None

    schema = User.model_json_schema()
    # Should not raise an error
    result = map_to_gemini_function_schema(schema)

    assert result["properties"]["age"]["nullable"] is True
    assert result["properties"]["email"]["nullable"] is True
    assert result["required"] == ["name"]


def test_union_types_rejected():
    """Test that Union types (not Optional) throw an error."""

    class UserWithUnion(BaseModel):
        name: str
        value: Union[int, str]  # Should be rejected

    schema = UserWithUnion.model_json_schema()

    with pytest.raises(ValueError, match="Union types"):
        map_to_gemini_function_schema(schema)


@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_STRUCTURED_OUTPUTS, instructor.Mode.GENAI_TOOLS]
)
def test_genai_api_call_with_different_types(mode):
    """Test actual API call with genai SDK using different types."""

    class UserProfile(BaseModel):
        name: str
        age: int
        email: Optional[str] = None
        is_premium: bool
        score: float

    client = instructor.from_provider("google/gemini-2.0-flash", mode=mode)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Create a user profile for John Doe, 25 years old, premium user with score 85.5",
            }
        ],
        response_model=UserProfile,
    )

    assert isinstance(response, UserProfile)
    assert response.name == "John Doe"
    assert response.email is None


@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_STRUCTURED_OUTPUTS, instructor.Mode.GENAI_TOOLS]
)
def test_genai_api_call_with_nested_models(mode):
    """Test API call with nested models (multiple users)."""

    class User(BaseModel):
        name: str
        age: int
        department: Optional[str] = None

    class UserList(BaseModel):
        users: list[User]

    client = instructor.from_provider("google/gemini-2.0-flash", mode=mode)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Create a list of 3 employees: Alice (30, Engineering), Bob (25, Marketing), Charlie (35)",
            }
        ],
        response_model=UserList,
    )

    assert isinstance(response, UserList)
    assert len(response.users) == 3
    assert {user.name for user in response.users} == {"Alice", "Bob", "Charlie"}
    assert {user.age for user in response.users} == {25, 30, 35}
    assert {user.department for user in response.users} == {
        None,
        "Engineering",
        "Marketing",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_STRUCTURED_OUTPUTS, instructor.Mode.GENAI_TOOLS]
)
async def test_genai_api_call_with_different_types_async(mode):
    """Test actual async API call with genai SDK using different types."""

    class UserProfile(BaseModel):
        name: str
        age: int
        email: Optional[str] = None
        is_premium: bool
        score: float

    client = instructor.from_provider(
        "google/gemini-2.0-flash", mode=mode, async_client=True
    )

    response = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Create a user profile for John Doe, 25 years old, premium user with score 85.5",
            }
        ],
        response_model=UserProfile,
    )

    assert isinstance(response, UserProfile)
    assert response.name == "John Doe"
    assert response.email is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_STRUCTURED_OUTPUTS, instructor.Mode.GENAI_TOOLS]
)
async def test_genai_api_call_with_nested_models_async(mode):
    """Test async API call with nested models (multiple users)."""

    class User(BaseModel):
        name: str
        age: int
        department: Optional[str] = None

    class UserList(BaseModel):
        users: list[User]

    client = instructor.from_provider(
        "google/gemini-2.0-flash", mode=mode, async_client=True
    )

    response = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Create a list of 3 employees: Alice (30, Engineering), Bob (25, Marketing), Charlie (35)",
            }
        ],
        response_model=UserList,
    )

    assert isinstance(response, UserList)
    assert len(response.users) == 3
    assert {user.name for user in response.users} == {"Alice", "Bob", "Charlie"}
    assert {user.age for user in response.users} == {25, 30, 35}
    assert {user.department for user in response.users} == {
        None,
        "Engineering",
        "Marketing",
    }
