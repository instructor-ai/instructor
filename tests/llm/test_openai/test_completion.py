import pytest
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
import instructor


class User(BaseModel):
    name: str
    age: int


def test_create_with_completion():
    client = OpenAI()
    client = instructor.from_openai(client)

    users, completion = client.chat.completions.create_with_completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, ChatCompletion)


@pytest.mark.asyncio
async def test_create_with_completion_async():
    client = AsyncOpenAI()
    client = instructor.from_openai(client)

    users, completion = await client.chat.completions.create_with_completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, ChatCompletion)
