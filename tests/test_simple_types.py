from instructor.dsl import is_simple_type, Partial
from pydantic import BaseModel


def test_enum_simple():
    from enum import Enum

    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    assert is_simple_type(Color), "Failed for type: " + str(Color)


def test_standard_types():
    for t in [str, int, float, bool]:
        assert is_simple_type(t), "Failed for type: " + str(t)


def test_partial_not_simple():
    class SampleModel(BaseModel):
        data: int

    assert not is_simple_type(Partial[SampleModel]), "Failed for type: " + str(
        Partial[int]
    )


def test_annotated_simple():
    from pydantic import Field
    from typing import Annotated

    new_type = Annotated[int, Field(description="test")]

    assert is_simple_type(new_type), "Failed for type: " + str(new_type)


def test_literal_simple():
    from typing import Literal

    new_type = Literal[1, 2, 3]

    assert is_simple_type(new_type), "Failed for type: " + str(new_type)


def test_union_simple():
    from typing import Union

    new_type = Union[int, str]

    assert is_simple_type(new_type), "Failed for type: " + str(new_type)


def test_iterable_not_simple():
    from typing import Iterable

    new_type = Iterable[int]

    assert not is_simple_type(new_type), "Failed for type: " + str(new_type)
