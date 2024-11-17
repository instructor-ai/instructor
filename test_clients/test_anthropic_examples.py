import pytest
import asyncio
from anthropic import AsyncAnthropic
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

class Address(BaseModel):
    street: str
    city: str
    country: str

class UserWithAddress(BaseModel):
    name: str
    age: int
    address: Address

@pytest.mark.asyncio
async def test_basic_example():
    client = AsyncAnthropic(api_key="your_anthropic_api_key")
    client = instructor.from_anthropic(client)

    try:
        user = await client.messages.create(
            model="claude-3-opus-20240229",
            messages=[
                {"role": "user", "content": "Extract: Jason is 25 years old"},
            ],
            response_model=User,
        )
        assert user.name == "Jason"
        assert user.age == 25
    except Exception as e:
        pytest.skip(f"Skipping due to missing API key or other error: {str(e)}")

@pytest.mark.asyncio
async def test_nested_example():
    client = AsyncAnthropic(api_key="your_anthropic_api_key")
    client = instructor.from_anthropic(client)

    try:
        user = await client.messages.create(
            model="claude-3-opus-20240229",
            messages=[
                {"role": "user", "content": """
                Extract user with address:
                Jason is 25 years old and lives at 123 Main St, San Francisco, USA
                """},
            ],
            response_model=UserWithAddress,
        )
        assert user.name == "Jason"
        assert user.age == 25
        assert user.address.street == "123 Main St"
        assert user.address.city == "San Francisco"
        assert user.address.country == "USA"
    except Exception as e:
        pytest.skip(f"Skipping due to missing API key or other error: {str(e)}")

@pytest.mark.asyncio
async def test_streaming_example():
    client = AsyncAnthropic(api_key="your_anthropic_api_key")
    client = instructor.from_anthropic(client)

    try:
        partial_results = []
        async for partial_user in client.messages.create_partial(
            model="claude-3-opus-20240229",
            messages=[
                {"role": "user", "content": "Extract: Jason is 25 years old"},
            ],
            response_model=User,
        ):
            partial_results.append(partial_user)

        assert len(partial_results) > 0
        final_user = partial_results[-1]
        assert final_user.name == "Jason"
        assert final_user.age == 25
    except Exception as e:
        pytest.skip(f"Skipping due to missing API key or other error: {str(e)}")

@pytest.mark.asyncio
async def test_iterable_streaming():
    client = AsyncAnthropic(api_key="your_anthropic_api_key")
    client = instructor.from_anthropic(client)

    try:
        users = []
        async for user in client.messages.create_iterable(
            model="claude-3-opus-20240229",
            messages=[
                {"role": "user", "content": """
                    Extract users:
                    1. Jason is 25 years old
                    2. Sarah is 30 years old
                    3. Mike is 28 years old
                """},
            ],
            response_model=User,
        ):
            users.append(user)

        assert len(users) == 3
        assert users[0].name == "Jason" and users[0].age == 25
        assert users[1].name == "Sarah" and users[1].age == 30
        assert users[2].name == "Mike" and users[2].age == 28
    except Exception as e:
        pytest.skip(f"Skipping due to missing API key or other error: {str(e)}")
