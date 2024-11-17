import sys
import os
import openai
import instructor
import pytest
from pydantic import BaseModel
from typing import List, Optional
import asyncio

def test_basic_example():
    print("Testing basic example...")
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    client = instructor.patch(client, mode=instructor.Mode.JSON)

    class User(BaseModel):
        name: str
        age: int

    try:
        user = client.chat.completions.create(
            model="llama2",  # Using available model
            messages=[
                {"role": "user", "content": "Extract: Jason is 25 years old"},
            ],
            response_model=User,
        )
        print(f"Basic test result: {user}")
        assert user.name == "Jason"
        assert user.age == 25
    except Exception as e:
        print(f"Error in basic test: {str(e)}")
        pytest.fail(f"Basic test failed: {str(e)}")

@pytest.mark.asyncio
async def test_async_example():
    print("\nTesting async example...")
    client = openai.AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    client = instructor.patch(client, mode=instructor.Mode.JSON)

    class User(BaseModel):
        name: str
        age: int

    try:
        user = await client.chat.completions.create(
            model="llama2",  # Using available model
            messages=[
                {"role": "user", "content": "Extract: Jason is 25 years old"},
            ],
            response_model=User,
        )
        print(f"Async test result: {user}")
        assert user.name == "Jason"
        assert user.age == 25
    except Exception as e:
        print(f"Error in async test: {str(e)}")
        pytest.fail(f"Async test failed: {str(e)}")

def test_nested_example():
    print("\nTesting nested example...")
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    client = instructor.patch(client, mode=instructor.Mode.JSON)

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class User(BaseModel):
        name: str
        age: int
        addresses: List[Address]

    try:
        user = client.chat.completions.create(
            model="llama2",  # Using available model
            messages=[
                {"role": "user", "content": """
                    Extract: Jason is 25 years old.
                    He lives at 123 Main St, New York, USA
                    and has a summer house at 456 Beach Rd, Miami, USA
                """},
            ],
            response_model=User,
        )
        print(f"Nested test result: {user}")
        assert user.name == "Jason"
        assert user.age == 25
        assert len(user.addresses) == 2
        assert user.addresses[0].city == "New York"
        assert user.addresses[1].city == "Miami"
    except Exception as e:
        print(f"Error in nested test: {str(e)}")
        pytest.fail(f"Nested test failed: {str(e)}")

def test_streaming_support():
    print("\nTesting streaming support...")
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    client = instructor.patch(client, mode=instructor.Mode.JSON)

    class User(BaseModel):
        name: str
        age: int

    try:
        # Test partial streaming
        for partial_user in client.chat.completions.create_partial(
            model="llama2",
            messages=[
                {"role": "user", "content": "Extract: Jason is 25 years old"},
            ],
            response_model=User,
        ):
            print(f"Partial result: {partial_user}")
            if hasattr(partial_user, 'name'):
                assert partial_user.name == "Jason"
            if hasattr(partial_user, 'age'):
                assert partial_user.age == 25
    except Exception as e:
        print(f"Error in streaming test: {str(e)}")
        pytest.fail(f"Streaming test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
