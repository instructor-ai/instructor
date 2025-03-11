import pytest
from writerai import Writer, AsyncWriter
from writerai.types import Chat
from pydantic import BaseModel
import instructor


class User(BaseModel):
    name: str
    age: int


def test_create_with_completion():
    client = Writer()
    client = instructor.from_writer(client)

    users, completion = client.chat.completions.create_with_completion(
        model="palmyra-x-004",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, Chat)


def test_create_with_completion_bool():
    client = Writer()
    client = instructor.from_writer(client)

    response, completion = client.chat.completions.create_with_completion(
        model="palmyra-x-004",
        messages=[{"role": "user", "content": "Is paris the capital of France?"}],
        response_model=bool,  # type: ignore
    )

    # Verify we got a valid response
    assert isinstance(response, bool)
    assert response is True

    assert isinstance(completion, Chat)


@pytest.mark.asyncio
async def test_create_with_completion_async():
    client = AsyncWriter()
    client = instructor.from_writer(client)

    users, completion = await client.chat.completions.create_with_completion(
        model="palmyra-x-004",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, Chat)
