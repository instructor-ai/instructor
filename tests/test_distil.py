from typing import Any, Callable, cast
import pytest
import instructor

from openai import OpenAI
from pydantic import BaseModel

from instructor.distil import (
    Instructions,
    format_function,
    get_signature_from_fn,
    is_return_type_base_model_or_instance,
)

client = instructor.patch(OpenAI())

instructions = Instructions(
    name="test_distil",
)


class SimpleModel(BaseModel):  # type: ignore[misc]
    data: int


def test_must_have_hint() -> None:
    with pytest.raises(AssertionError):

        @instructions.distil
        def test_func(x: int):  # type: ignore[no-untyped-def]
            return SimpleModel(data=x)


def test_must_be_base_model() -> None:
    with pytest.raises(AssertionError):

        @instructions.distil
        def test_func(x: int) -> int:
            return SimpleModel(data=x)


def test_is_return_type_base_model_or_instance() -> None:
    def valid_function() -> SimpleModel:
        return SimpleModel(data=1)

    def invalid_function() -> int:
        return 1

    assert is_return_type_base_model_or_instance(valid_function)
    assert not is_return_type_base_model_or_instance(invalid_function)


def test_get_signature_from_fn() -> None:
    def test_function(a: int, b: str) -> float:  # type: ignore[empty-body]
        """Sample docstring"""
        pass

    result = get_signature_from_fn(test_function)
    expected = "def test_function(a: int, b: str) -> float"
    assert expected in result
    assert "Sample docstring" in result


def test_format_function() -> None:
    def sample_function(x: int) -> SimpleModel:
        """This is a docstring."""
        return SimpleModel(data=x)

    formatted = format_function(sample_function)
    assert "def sample_function(x: int) -> SimpleModel:" in formatted
    assert '"""This is a docstring."""' in formatted
    assert "return SimpleModel(data=x)" in formatted


def test_distil_decorator_without_arguments() -> None:
    @instructions.distil
    def test_func(x: int) -> SimpleModel:
        return SimpleModel(data=x)

    casted_test_func = cast(Callable[[int], SimpleModel], test_func)
    result: SimpleModel = casted_test_func(42)
    assert result.data == 42


def test_distil_decorator_with_name_argument() -> None:
    @instructions.distil(name="custom_name")
    def another_test_func(x: int) -> SimpleModel:
        return SimpleModel(data=x)

    casted_another_test_func = cast(Callable[[int], SimpleModel], another_test_func)
    result: SimpleModel = casted_another_test_func(55)
    assert result.data == 55


# Mock track function for decorator tests
def mock_track(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> None:
    pass


def fn(a: int, b: int) -> int:
    return client.chat.completions.create(
        messages=[], model="davinci", response_model=SimpleModel
    )
