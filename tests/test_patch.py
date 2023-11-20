import functools
import pytest
import instructor

from pydantic import BaseModel, ValidationError, BeforeValidator
from openai import OpenAI, AsyncOpenAI
from instructor import llm_validator
from typing_extensions import Annotated


from instructor.patch import is_async, wrap_chatcompletion, OVERRIDE_DOCS


def test_patch_completes_successfully():
    instructor.patch(OpenAI())


def test_apatch_completes_successfully():
    instructor.apatch(AsyncOpenAI())


@pytest.mark.asyncio
async def test_wrap_chatcompletion_wraps_async_input_function():
    async def input_function(*args, **kwargs):
        return "Hello, World!"

    wrapped_function = wrap_chatcompletion(input_function)
    result = await wrapped_function()

    assert result == "Hello, World!"


def test_wrap_chatcompletion_wraps_input_function():
    def input_function(*args, **kwargs):
        return "Hello, World!"

    wrapped_function = wrap_chatcompletion(input_function)
    result = wrapped_function()

    assert result == "Hello, World!"


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


def test_override_docs():
    assert (
        "response_model" in OVERRIDE_DOCS
    ), "response_model should be in OVERRIDE_DOCS"
