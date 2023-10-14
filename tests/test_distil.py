import pytest
import openai
from pydantic import BaseModel
from instructor.distil import (
    Instructions,
    format_function,
    get_signature_from_fn,
    is_return_type_base_model_or_instance,
)

# Replace `your_module_name` with your actual module name

instructions = Instructions(
    name="test_distil",
)


class SimpleModel(BaseModel):
    data: int


def test_must_have_hint():
    with pytest.raises(AssertionError):

        @instructions.distil
        def test_func(x: int):
            return SimpleModel(data=x)


def test_must_be_base_model():
    with pytest.raises(AssertionError):

        @instructions.distil
        def test_func(x) -> int:
            return SimpleModel(data=x)


def test_is_return_type_base_model_or_instance():
    def valid_function() -> SimpleModel:
        return SimpleModel(data=1)

    def invalid_function() -> int:
        return 1

    assert is_return_type_base_model_or_instance(valid_function)
    assert not is_return_type_base_model_or_instance(invalid_function)


def test_get_signature_from_fn():
    def test_function(a: int, b: str) -> float:
        """Sample docstring"""
        pass

    result = get_signature_from_fn(test_function)
    expected = "def test_function(a: int, b: str) -> float"
    assert expected in result
    assert "Sample docstring" in result


def test_format_function():
    def sample_function(x: int) -> SimpleModel:
        """This is a docstring."""
        return SimpleModel(data=x)

    formatted = format_function(sample_function)
    assert "def sample_function(x: int) -> SimpleModel:" in formatted
    assert '"""This is a docstring."""' in formatted
    assert "return SimpleModel(data=x)" in formatted


def test_distil_decorator_without_arguments():
    @instructions.distil
    def test_func(x: int) -> SimpleModel:
        return SimpleModel(data=x)

    result = test_func(42)
    assert result.data == 42


def test_distil_decorator_with_name_argument():
    @instructions.distil(name="custom_name")
    def another_test_func(x: int) -> SimpleModel:
        return SimpleModel(data=x)

    result = another_test_func(55)
    assert result.data == 55


# Mock track function for decorator tests
def mock_track(*args, **kwargs):
    pass


def fn(a: int, b: int) -> int:
    return openai.ChatCompletion.create(
        messages=[],
        model="davinci",
        response_model=SimpleModel,
    )
