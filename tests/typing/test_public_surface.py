from __future__ import annotations

# mypy: disable-error-code=unused-coroutine

from collections.abc import AsyncGenerator, Coroutine, Generator
from typing import Any, assert_type

import openai
from pydantic import BaseModel

from instructor.v2.core.client import AsyncInstructor, AsyncResponse, Instructor, Response
from instructor.v2.core.function_calls import response_schema
from instructor.v2.providers.genai.client import from_genai
from instructor.v2.providers.openai.client import (
    from_anyscale,
    from_databricks,
    from_deepseek,
    from_together,
)


class User(BaseModel):
    name: str


def check_response_helpers(sync_response: Response, async_response: AsyncResponse) -> None:
    assert_type(sync_response.create(response_model=User), User)
    assert_type(sync_response.create(response_model=None), Any)
    assert_type(sync_response.create_with_completion(response_model=User), tuple[User, Any])
    assert_type(sync_response.create_iterable(response_model=User), Generator[User, None, None])
    assert_type(sync_response.create_partial(response_model=User), Generator[User, None, None])

    create_coro = async_response.create(response_model=User)
    create_any_coro = async_response.create(response_model=None)
    completion_coro = async_response.create_with_completion(response_model=User)
    iterable_coro = async_response.create_iterable(response_model=User)

    assert_type(create_coro, Coroutine[Any, Any, User])
    assert_type(create_any_coro, Coroutine[Any, Any, Any])
    assert_type(
        completion_coro,
        Coroutine[Any, Any, tuple[User, Any]],
    )
    assert_type(
        iterable_coro,
        Coroutine[Any, Any, AsyncGenerator[User, None]],
    )


def check_openai_compatible_factories(
    sync_client: openai.OpenAI,
    async_client: openai.AsyncOpenAI,
) -> None:
    assert_type(from_anyscale("model"), Instructor)
    assert_type(from_anyscale("model", async_client=True), AsyncInstructor)
    assert_type(from_anyscale(sync_client), Instructor)
    assert_type(from_anyscale(async_client), AsyncInstructor)

    assert_type(from_together("model"), Instructor)
    assert_type(from_together("model", async_client=True), AsyncInstructor)
    assert_type(from_together(sync_client), Instructor)
    assert_type(from_together(async_client), AsyncInstructor)

    assert_type(from_databricks("model"), Instructor)
    assert_type(from_databricks("model", async_client=True), AsyncInstructor)
    assert_type(from_databricks(sync_client), Instructor)
    assert_type(from_databricks(async_client), AsyncInstructor)

    assert_type(from_deepseek("model"), Instructor)
    assert_type(from_deepseek("model", async_client=True), AsyncInstructor)
    assert_type(from_deepseek(sync_client), Instructor)
    assert_type(from_deepseek(async_client), AsyncInstructor)


def check_genai_factory(client: Any) -> None:
    assert_type(from_genai(client), Instructor)
    assert_type(from_genai(client, use_async=True), AsyncInstructor)


def check_base_model_helpers() -> None:
    assert_type(response_schema(User), type[User])
