import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types.message import Message
from pydantic import BaseModel
import instructor


class User(BaseModel):
    name: str
    age: int


def test_create_with_completion():
    client = Anthropic()
    client = instructor.from_anthropic(client)

    users, completion = client.messages.create_with_completion(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
        max_tokens=1000,
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, Message)


def test_create_with_completion_bool():
    client = Anthropic()
    client = instructor.from_anthropic(client)

    response, completion = client.messages.create_with_completion(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "Is paris the capital of France?"}],
        response_model=bool,  # type: ignore
        max_tokens=1000,
    )

    # Verify we got a valid response
    assert isinstance(response, bool)
    assert response is True

    assert isinstance(completion, Message)


@pytest.mark.asyncio
async def test_create_with_completion_async():
    client = AsyncAnthropic()
    client = instructor.from_anthropic(client)

    users, completion = await client.messages.create_with_completion(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
        max_tokens=1000,
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(
        completion, Message
    ), f"Completion is not a Message: {completion}, it is of type {type(completion)}"
