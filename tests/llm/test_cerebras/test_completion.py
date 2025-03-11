import pytest
from cerebras.cloud.sdk import Cerebras, AsyncCerebras
from cerebras.cloud.sdk.types.chat import ChatCompletion
from pydantic import BaseModel
import instructor


class User(BaseModel):
    name: str
    age: int


def test_create_with_completion():
    client = Cerebras()
    client = instructor.from_cerebras(client)

    users, completion = client.chat.completions.create_with_completion(
        model="llama-3.3-70b",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, ChatCompletion)


def test_create_with_completion_bool():
    client = Cerebras()
    client = instructor.from_cerebras(client)

    response, completion = client.chat.completions.create_with_completion(
        model="llama-3.3-70b",
        messages=[{"role": "user", "content": "Is paris the capital of France?"}],
        response_model=bool,  # type: ignore
    )

    # Verify we got a valid response
    assert isinstance(response, bool)
    assert response is True

    assert isinstance(completion, ChatCompletion)


@pytest.mark.asyncio
async def test_create_with_completion_async():
    client = AsyncCerebras()
    client = instructor.from_cerebras(client)

    users, completion = await client.chat.completions.create_with_completion(
        model="llama-3.3-70b",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, ChatCompletion)
