import os
import openai
import instructor
import anthropic
from pydantic import BaseModel
import pytest


class User(BaseModel):
    name: str
    age: int


def test_client_create():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    user = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


def test_client_messages_create():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    user = client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


def test_client_chat_completions_create_with_response():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    user, completion = client.chat.completions.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10

    from openai.types.chat import ChatCompletion

    assert isinstance(completion, ChatCompletion)


def test_client_chat_completions_create():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    user = client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


def test_client_chat_completions_create_partial():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    for user in client.chat.completions.create_partial(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    ):
        assert isinstance(user, User)


def test_client_chat_completions_create_iterable():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    users = [
        user
        for user in client.chat.completions.create_iterable(
            response_model=User,
            messages=[{"role": "user", "content": "Alice is 25, Bob is 30"}],
            temperature=0,
        )
    ]
    assert len(users) == 2


@pytest.mark.asyncio
async def test_async_client_chat_completions_create():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-3.5-turbo")

    user = await instructor_client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.asyncio
async def test_async_client_chat_completions_create_partial():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-3.5-turbo")

    async for user in instructor_client.chat.completions.create_partial(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    ):
        assert isinstance(user, User)


@pytest.mark.asyncio
async def test_async_client_chat_completions_create_iterable():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-3.5-turbo")

    async for user in instructor_client.chat.completions.create_iterable(
        response_model=User,
        messages=[{"role": "user", "content": "Alice is 25, Bob is 30"}],
        temperature=0,
    ):
        assert isinstance(user, User)


@pytest.mark.asyncio
async def test_async_client_chat_completions_create_with_response():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-3.5-turbo")

    user, response = await instructor_client.chat.completions.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    from openai.types.chat import ChatCompletion

    assert user.name == "Jason"
    assert user.age == 10
    assert isinstance(response, ChatCompletion)


def test_client_from_anthropic_with_response():
    client = instructor.from_anthropic(
        anthropic.Anthropic(),
        max_tokens=1000,
        model="claude-3-haiku-20240307",
    )

    user, response = client.messages.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10
    assert isinstance(response, anthropic.types.Message)


def test_client_anthropic_response():
    client = anthropic.Anthropic()
    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="claude-3-haiku-20240307",
    )

    user = instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.skip(reason="Skip for now")
def test_client_anthropic_bedrock_response():
    client = anthropic.AnthropicBedrock(
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        aws_region=os.getenv("AWS_REGION_NAME"),
    )

    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )

    user = instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.asyncio
async def test_async_client_anthropic_response():
    client = anthropic.AsyncAnthropic()
    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="claude-3-haiku-20240307",
    )

    user = await instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.skip(reason="Skip for now")
@pytest.mark.asyncio
async def test_async_client_anthropic_bedrock_response():
    client = anthropic.AsyncAnthropicBedrock(
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        aws_region=os.getenv("AWS_REGION_NAME"),
    )

    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )

    user = await instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10
