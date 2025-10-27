"""
Integration tests for create_with_completion with List[T] and Iterable[T] (#1303, #1305)
"""

from collections.abc import Iterable

import pytest
from pydantic import BaseModel

import instructor
from instructor.dsl import ListResponse
from openai import AsyncOpenAI, OpenAI


class User(BaseModel):
    """Test user model"""

    name: str
    age: int


class TestIterableCreateWithCompletion:
    """Test create_with_completion with Iterable response models"""

    @pytest.mark.vcr
    def test_create_with_completion_list_sync(self):
        """Test create_with_completion with List[User] in sync mode"""
        client = instructor.from_openai(OpenAI())

        users, completion = client.chat.completions.create_with_completion(
            model="gpt-4o-mini",
            response_model=list[User],
            messages=[
                {
                    "role": "user",
                    "content": "Extract users: John is 30 years old, Jane is 25 years old",
                }
            ],
        )

        # Verify response
        assert isinstance(users, ListResponse)
        assert len(users) == 2
        assert users[0].name.lower() == "john"
        assert users[0].age == 30
        assert users[1].name.lower() == "jane"
        assert users[1].age == 25

        # Verify raw response is attached
        assert hasattr(users, "_raw_response")
        assert users._raw_response is not None
        assert users._raw_response == completion

        # Verify completion object
        assert completion is not None
        assert hasattr(completion, "usage")
        assert completion.usage.total_tokens > 0

    @pytest.mark.vcr
    def test_create_with_completion_list_works_like_list(self):
        """Verify ListResponse behaves like a normal list"""
        client = instructor.from_openai(OpenAI())

        users, completion = client.chat.completions.create_with_completion(
            model="gpt-4o-mini",
            response_model=list[User],
            messages=[
                {
                    "role": "user",
                    "content": "Extract users: Alice is 28, Bob is 35",
                }
            ],
        )

        # List operations should work
        assert len(users) >= 1
        assert users[0] is not None
        assert isinstance(users[0], User)

        # Iteration should work
        for user in users:
            assert isinstance(user, User)
            assert user.name is not None
            assert user.age > 0

        # Append should work and preserve raw response
        users.append(User(name="Carol", age=32))
        assert len(users) >= 2
        assert users._raw_response is not None

    @pytest.mark.vcr
    def test_create_with_completion_iterable_sync(self):
        """Test create_with_completion with Iterable[User] in sync mode"""
        client = instructor.from_openai(OpenAI())

        users, completion = client.chat.completions.create_with_completion(
            model="gpt-4o-mini",
            response_model=Iterable[User],
            messages=[
                {
                    "role": "user",
                    "content": "Extract users: David is 40, Eve is 35",
                }
            ],
        )

        # Verify response
        assert isinstance(users, ListResponse)
        assert len(users) >= 1

        # Verify items are User instances
        for user in users:
            assert isinstance(user, User)

        # Verify raw response is attached
        assert hasattr(users, "_raw_response")
        assert users._raw_response is not None

        # Verify completion object
        assert completion is not None
        assert hasattr(completion, "usage")

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_create_with_completion_list_async(self):
        """Test create_with_completion with List[User] in async mode"""
        client = instructor.from_openai(AsyncOpenAI())

        users, completion = await client.chat.completions.create_with_completion(
            model="gpt-4o-mini",
            response_model=list[User],
            messages=[
                {
                    "role": "user",
                    "content": "Extract users: Frank is 45, Grace is 38",
                }
            ],
        )

        # Verify response
        assert isinstance(users, ListResponse)
        assert len(users) >= 1

        # Verify items
        for user in users:
            assert isinstance(user, User)

        # Verify raw response
        assert hasattr(users, "_raw_response")
        assert users._raw_response is not None
        assert completion is not None

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_create_with_completion_iterable_async(self):
        """Test create_with_completion with Iterable[User] in async mode"""
        client = instructor.from_openai(AsyncOpenAI())

        users, completion = await client.chat.completions.create_with_completion(
            model="gpt-4o-mini",
            response_model=Iterable[User],
            messages=[
                {
                    "role": "user",
                    "content": "Extract users: Henry is 50, Iris is 42",
                }
            ],
        )

        # Verify response is ListResponse
        assert isinstance(users, ListResponse)

        # Verify raw response
        assert hasattr(users, "_raw_response")
        assert users._raw_response is not None
        assert completion is not None


class TestListResponseRawResponseAccess:
    """Test raw response accessibility (#1305)"""

    @pytest.mark.vcr
    def test_raw_response_via_attribute(self):
        """Test accessing raw response via ._raw_response attribute"""
        client = instructor.from_openai(OpenAI())

        users, completion = client.chat.completions.create_with_completion(
            model="gpt-4o-mini",
            response_model=list[User],
            messages=[
                {
                    "role": "user",
                    "content": "Extract users: Jack is 55, Kate is 48",
                }
            ],
        )

        # Both should give same raw response
        assert users._raw_response == completion
        assert hasattr(completion, "usage")
        assert completion.usage.total_tokens > 0

    @pytest.mark.vcr
    def test_raw_response_via_getter(self):
        """Test accessing raw response via get_raw_response() method"""
        client = instructor.from_openai(OpenAI())

        users, completion = client.chat.completions.create_with_completion(
            model="gpt-4o-mini",
            response_model=list[User],
            messages=[
                {
                    "role": "user",
                    "content": "Extract users: Leo is 60, Maya is 52",
                }
            ],
        )

        # get_raw_response() should work
        raw = users.get_raw_response()
        assert raw is not None
        assert raw == completion


class TestBackwardCompatibility:
    """Test backward compatibility with existing code"""

    @pytest.mark.vcr
    def test_single_model_create_with_completion_still_works(self):
        """Verify single model create_with_completion still works"""
        client = instructor.from_openai(OpenAI())

        user, completion = client.chat.completions.create_with_completion(
            model="gpt-4o-mini",
            response_model=User,
            messages=[
                {
                    "role": "user",
                    "content": "Extract user: Noah is 65",
                }
            ],
        )

        # Single model should work as before
        assert isinstance(user, User)
        assert hasattr(user, "_raw_response")
        assert user._raw_response == completion
        assert completion is not None
