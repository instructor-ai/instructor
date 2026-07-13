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
