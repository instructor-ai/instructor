from typing import Optional, Generator
import openai
import instructor
from pydantic import BaseModel
import pytest
import os
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice, ChatCompletionMessage

# Load environment variables from .env.tests
load_dotenv(".env.tests")

class User(BaseModel):
    name: str
    age: int

def test_anyscale_basic() -> None:
    """Test basic Anyscale functionality"""
    api_key = os.getenv("ANYSCALE_API_KEY")
    if api_key == "missing":
        pytest.skip("Anyscale API key not available")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.endpoints.anyscale.com/v1"
    )
    client = instructor.patch(client)

    try:
        user = client.chat.completions.create(
            model="meta-llama/Llama-2-70b-chat-hf",
            messages=[
                {"role": "user", "content": "Extract: Jason is 25 years old"},
            ],
            response_model=User,
        )
        assert user.name == "Jason"
        assert user.age == 25
    except Exception as e:
        pytest.fail(f"Basic test failed: {str(e)}")

def test_anyscale_streaming() -> None:
    """Test Anyscale streaming capabilities"""
    api_key = os.getenv("ANYSCALE_API_KEY")
    if api_key == "missing":
        pytest.skip("Anyscale API key not available")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.endpoints.anyscale.com/v1"
    )
    client = instructor.patch(client)

    class UserWithBio(BaseModel):
        name: str
        age: int
        bio: str

    try:
        stream_success = False
        for partial in client.chat.completions.create_partial(
            model="meta-llama/Llama-2-70b-chat-hf",
            messages=[
                {"role": "user", "content": "Create a user profile for Jason, age 25"},
            ],
            response_model=UserWithBio,
        ):
            if partial:
                stream_success = True
                break

        assert stream_success, "Streaming did not produce any partial results"
    except Exception as e:
        pytest.fail(f"Streaming test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])
