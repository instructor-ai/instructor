import functools

from openai import AsyncOpenAI, OpenAI
import pytest

import instructor
from instructor.utils import is_async


def test_patch_completes_successfully():
    with OpenAI(api_key="test-key") as client:
        instructor.patch(client)


@pytest.mark.asyncio
async def test_apatch_completes_successfully():
    async with AsyncOpenAI(api_key="test-key") as client:
        with pytest.warns(
            DeprecationWarning, match="apatch is deprecated, use patch instead"
        ):
            instructor.apatch(client)


def test_is_async_returns_true_if_function_is_async():
    async def async_function():
        pass

    assert is_async(async_function) is True


def test_is_async_returns_false_if_function_is_not_async():
    def sync_function():
        pass

    assert is_async(sync_function) is False


def test_is_async_returns_true_if_wrapped_function_is_async():
    async def async_function():
        pass

    @functools.wraps(async_function)
    def wrapped_function():
        pass

    assert is_async(wrapped_function) is True


def test_is_async_returns_true_if_double_wrapped_function_is_async():
    async def async_function():
        pass

    @functools.wraps(async_function)
    def wrapped_function():
        pass

    @functools.wraps(wrapped_function)
    def double_wrapped_function():
        pass

    assert is_async(double_wrapped_function) is True


def test_is_async_returns_true_if_triple_wrapped_function_is_async():
    async def async_function():
        pass

    @functools.wraps(async_function)
    def wrapped_function():
        pass

    @functools.wraps(wrapped_function)
    def double_wrapped_function():
        pass

    @functools.wraps(double_wrapped_function)
    def triple_wrapped_function():
        pass

    assert is_async(triple_wrapped_function) is True


class AsyncCallable:
    async def __call__(self, **_kwargs):
        return {"choices": [{"message": {"content": "{}"}}]}


class SyncCallable:
    def __call__(self, **_kwargs):
        return {"choices": [{"message": {"content": "{}"}}]}


@pytest.mark.parametrize(
    "callable_obj,expected_is_async",
    [
        (AsyncCallable(), True),
        (SyncCallable(), False),
    ],
)
def test_is_async_callable_instances(callable_obj, expected_is_async):
    assert is_async(callable_obj) is expected_is_async


@pytest.mark.parametrize(
    "callable_obj,expected_is_async",
    [
        (AsyncCallable(), True),
        (SyncCallable(), False),
    ],
)
def test_is_async_wrapped_callable_instances(callable_obj, expected_is_async):
    @functools.wraps(callable_obj)
    def wrapped(*args, **kwargs):
        return callable_obj(*args, **kwargs)

    assert is_async(wrapped) is expected_is_async


def test_patch_async_callable_instance_routes_to_async():
    import inspect

    create = AsyncCallable()
    patched = instructor.patch(create=create, mode=instructor.Mode.JSON)
    assert inspect.iscoroutinefunction(patched) is True


@pytest.mark.asyncio
async def test_patch_async_callable_instance_execution():
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str

    class AsyncMockCreate:
        async def __call__(self, *_args, **_kwargs):
            return ChatCompletion(
                id="test-123",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            content='{"name": "Ada"}',
                            role="assistant",
                        ),
                    )
                ],
                created=1234567890,
                model="gpt-4o",
                object="chat.completion",
            )

    create = AsyncMockCreate()
    patched = instructor.patch(create=create, mode=instructor.Mode.JSON)
    res = await patched(
        response_model=Person,
        messages=[{"role": "user", "content": "Extract Ada"}],
    )
    assert isinstance(res, Person)
    assert res.name == "Ada"
