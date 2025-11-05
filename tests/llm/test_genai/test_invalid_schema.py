import pytest
from typing import Optional, Union

import instructor
from pydantic import BaseModel
from .util import models, modes
from itertools import product
from instructor.providers.gemini.utils import map_to_gemini_function_schema


@pytest.mark.parametrize("mode,model", product(modes, models))
def test_nested(mode, model):
    """Test that nested schemas are supported."""
    client = instructor.from_provider(f"google/{model}", mode=mode)

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Optional[Address] = None

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "John loves to go gardenning with his friends",
            }
        ],
        response_model=Person,
    )

    assert resp.name == "John"  # type: ignore
    assert resp.address is None  # type: ignore


@pytest.mark.parametrize("mode,model", product(modes, models))
def test_union(mode, model):
    """Test union type behavior with Gemini.
    
    GENAI_STRUCTURED_OUTPUTS mode now supports Union types via response_json_schema
    (when the SDK supports it), while GENAI_TOOLS mode still rejects them.
    """
    client = instructor.from_provider(f"google/{model}", mode=mode)

    class UserData(BaseModel):
        name: str
        id_value: Union[str, int]

    if mode == instructor.Mode.GENAI_STRUCTURED_OUTPUTS:
        from google.genai import types
        supports_json_schema = hasattr(types.GenerateContentConfig, "__annotations__") and (
            "response_json_schema" in types.GenerateContentConfig.__annotations__
        )
        
        if supports_json_schema:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "User name is Alice with ID 12345"}],
                response_model=UserData,
            )
            assert response.name == "Alice"
            assert response.id_value in [12345, "12345"]
        else:
            with pytest.raises(
                ValueError,
                match=r"Gemini does not support Union types \(except Optional\)\. Please change your function schema",
            ):
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "User name is Alice with ID 12345"}],
                    response_model=UserData,
                )
    else:
        with pytest.raises(
            ValueError,
            match=r"Gemini does not support Union types \(except Optional\)\. Please change your function schema",
        ):
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "User name is Alice with ID 12345"}],
                response_model=UserData,
            )


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


def test_union_types_rejected_schema():
    """Test that Union types (not Optional) throw an error in schema mapping."""

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

    client = instructor.from_provider("google/gemini-2.5-flash", mode=mode)

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

    client = instructor.from_provider("google/gemini-2.5-flash", mode=mode)

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
        "google/gemini-2.5-flash", mode=mode, async_client=True
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
        "google/gemini-2.5-flash", mode=mode, async_client=True
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


@pytest.mark.parametrize("model", ["gemini-2.5-flash"])
def test_union_with_multiple_variants(model):
    """Test Union types with multiple variants (3+) in GENAI_STRUCTURED_OUTPUTS mode."""
    from google.genai import types
    
    supports_json_schema = hasattr(types.GenerateContentConfig, "__annotations__") and (
        "response_json_schema" in types.GenerateContentConfig.__annotations__
    )
    
    if not supports_json_schema:
        pytest.skip("response_json_schema not supported in this SDK version")
    
    from typing import Literal
    
    class TextContent(BaseModel):
        type: Literal["text"] = "text"
        content: str
    
    class ImageContent(BaseModel):
        type: Literal["image"] = "image"
        url: str
        alt_text: Optional[str] = None
    
    class VideoContent(BaseModel):
        type: Literal["video"] = "video"
        url: str
        duration: int
    
    class MediaPost(BaseModel):
        title: str
        media: Union[TextContent, ImageContent, VideoContent]
    
    client = instructor.from_provider(
        f"google/{model}",
        mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Create a post with title 'My Vacation' and an image at https://example.com/photo.jpg",
            }
        ],
        response_model=MediaPost,
    )
    
    assert response.title == "My Vacation"
    assert isinstance(response.media, ImageContent)
    assert response.media.url == "https://example.com/photo.jpg"


@pytest.mark.parametrize("model", ["gemini-2.5-flash"])
def test_union_with_nested_objects(model):
    """Test Union types with nested objects in GENAI_STRUCTURED_OUTPUTS mode."""
    from google.genai import types
    
    supports_json_schema = hasattr(types.GenerateContentConfig, "__annotations__") and (
        "response_json_schema" in types.GenerateContentConfig.__annotations__
    )
    
    if not supports_json_schema:
        pytest.skip("response_json_schema not supported in this SDK version")
    
    from typing import Literal
    
    class Address(BaseModel):
        street: str
        city: str
        country: str
    
    class PhysicalLocation(BaseModel):
        type: Literal["physical"] = "physical"
        address: Address
    
    class VirtualLocation(BaseModel):
        type: Literal["virtual"] = "virtual"
        url: str
        platform: str
    
    class Event(BaseModel):
        name: str
        location: Union[PhysicalLocation, VirtualLocation]
    
    client = instructor.from_provider(
        f"google/{model}",
        mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Create an event called 'Tech Conference' at 123 Main St, San Francisco, USA",
            }
        ],
        response_model=Event,
    )
    
    assert response.name == "Tech Conference"
    assert isinstance(response.location, PhysicalLocation)
    assert response.location.address.city == "San Francisco"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gemini-2.5-flash"])
async def test_union_with_retry_async(model):
    """Test Union types with retry/reask mechanism in GENAI_STRUCTURED_OUTPUTS mode."""
    from google.genai import types
    
    supports_json_schema = hasattr(types.GenerateContentConfig, "__annotations__") and (
        "response_json_schema" in types.GenerateContentConfig.__annotations__
    )
    
    if not supports_json_schema:
        pytest.skip("response_json_schema not supported in this SDK version")
    
    from typing import Literal
    from pydantic import field_validator
    
    class SpamContent(BaseModel):
        type: Literal["spam"] = "spam"
        reason: str
        
        @field_validator("reason")
        @classmethod
        def reason_must_be_detailed(cls, v):
            if len(v) < 10:
                raise ValueError("Reason must be at least 10 characters")
            return v
    
    class SafeContent(BaseModel):
        type: Literal["safe"] = "safe"
        summary: str
    
    class ModerationResult(BaseModel):
        decision: Union[SpamContent, SafeContent]
    
    client = instructor.from_provider(
        f"google/{model}",
        mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
        async_client=True,
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Moderate this: 'Buy cheap watches now!'",
            }
        ],
        response_model=ModerationResult,
        max_retries=2,
    )
    
    assert isinstance(response.decision, (SpamContent, SafeContent))
    if isinstance(response.decision, SpamContent):
        assert len(response.decision.reason) >= 10
