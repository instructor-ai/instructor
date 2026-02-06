import unittest
from instructor.dsl.simple_type import is_simple_type
from pydantic import BaseModel
from enum import Enum
import typing


class SimpleTypeTests(unittest.TestCase):
    def test_is_simple_type_with_base_model(self):
        class MyModel(BaseModel):
            label: str

        self.assertFalse(is_simple_type(MyModel))

    def test_is_simple_type_with_str(self):
        self.assertTrue(is_simple_type(str))

    def test_is_simple_type_with_int(self):
        self.assertTrue(is_simple_type(int))

    def test_is_simple_type_with_float(self):
        self.assertTrue(is_simple_type(float))

    def test_is_simple_type_with_bool(self):
        self.assertTrue(is_simple_type(bool))

    def test_is_simple_type_with_enum(self):
        class MyEnum(Enum):
            VALUE = 1

        self.assertTrue(is_simple_type(MyEnum))

    def test_is_simple_type_with_annotated(self):
        AnnotatedType = typing.Annotated[int, "example"]
        self.assertTrue(is_simple_type(AnnotatedType))

    def test_is_simple_type_with_literal(self):
        LiteralType = typing.Literal[1, 2, 3]
        self.assertTrue(is_simple_type(LiteralType))

    def test_is_simple_type_with_union(self):
        UnionType = typing.Union[int, str]
        self.assertTrue(is_simple_type(UnionType))

    def test_is_simple_type_with_iterable(self):
        IterableType = typing.Iterable[int]
        self.assertFalse(is_simple_type(IterableType))


if __name__ == "__main__":
    unittest.main()
