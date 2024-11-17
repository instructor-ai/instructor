import os
from openai import OpenAI
import instructor
from pydantic import BaseModel
import pytest
from typing import List

# Enable instructor patches for OpenAI client
client = instructor.patch(OpenAI())

class User(BaseModel):
    name: str
    age: int

class Address(BaseModel):
    street: str
    city: str
    country: str

class UserWithAddresses(BaseModel):
    name: str
    age: int
    addresses: List[Address]

def test_sync_example():
    """Test basic synchronous extraction"""
    try:
        user = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Extract: Jason is 25 years old"},
            ],
            response_model=User,
        )
        assert isinstance(user, User)
        assert user.name == "Jason"
        assert user.age == 25
    except Exception as e:
        pytest.fail(f"Sync example failed: {str(e)}")

def test_nested_example():
    """Test nested object extraction"""
    try:
        user = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": """
                    Extract: Jason is 25 years old.
                    He lives at 123 Main St, New York, USA
                    and has a summer house at 456 Beach Rd, Miami, USA
                """},
            ],
            response_model=UserWithAddresses,
        )
        assert isinstance(user, UserWithAddresses)
        assert user.name == "Jason"
        assert user.age == 25
        assert len(user.addresses) == 2
        assert user.addresses[0].city == "New York"
        assert user.addresses[1].city == "Miami"
    except Exception as e:
        pytest.fail(f"Nested example failed: {str(e)}")

def test_streaming_example():
    """Test streaming functionality"""
    try:
        partial_users = []
        for partial_user in client.chat.completions.create_partial(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Create a user profile for Jason, age 25"},
            ],
            response_model=User,
        ):
            assert isinstance(partial_user, User)
            partial_users.append(partial_user)

        # Verify we got streaming updates
        assert len(partial_users) > 0
        final_user = partial_users[-1]
        assert final_user.name == "Jason"
        assert final_user.age == 25
    except Exception as e:
        pytest.fail(f"Streaming example failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])
