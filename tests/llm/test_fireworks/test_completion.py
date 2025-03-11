import pytest
from fireworks.client import Fireworks, AsyncFireworks
from fireworks.client.chat_completion import ChatCompletionResponse
from pydantic import BaseModel
import instructor
from .util import modes


class User(BaseModel):
    name: str
    age: int


def test_create_with_completion():
    client = Fireworks()
    client = instructor.from_fireworks(client)

    users, completion = client.chat.completions.create_with_completion(
        model="accounts/fireworks/models/llama-v3-8b-instruct",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, ChatCompletionResponse)


def test_create_with_completion_bool():
    client = Fireworks()
    client = instructor.from_fireworks(client)

    response, completion = client.chat.completions.create_with_completion(
        model="accounts/fireworks/models/llama-v3-8b-instruct",
        messages=[{"role": "user", "content": "Is paris the capital of France?"}],
        response_model=bool,  # type: ignore
    )

    # Verify we got a valid response
    assert isinstance(response, bool)
    assert response is True

    assert isinstance(completion, ChatCompletionResponse)


@pytest.mark.asyncio
async def test_create_with_completion_async():
    client = AsyncFireworks()
    client = instructor.from_fireworks(client)

    users, completion = await client.chat.completions.create_with_completion(
        model="accounts/fireworks/models/llama-v3-8b-instruct",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, ChatCompletionResponse)
