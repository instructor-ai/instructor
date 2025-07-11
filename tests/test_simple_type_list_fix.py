import unittest
import sys
from typing import Union
from instructor.dsl.simple_type import is_simple_type
from pydantic import BaseModel


class TestUser(BaseModel):
    name: str
    age: int


class TestSimpleTypeListFix(unittest.TestCase):
    def test_list_with_basemodel_is_not_simple(self):
        """Test that list[BaseModel] is correctly identified as NOT a simple type."""
        response_model = list[TestUser]
        self.assertFalse(
            is_simple_type(response_model),
            "list[BaseModel] should NOT be a simple type",
        )

    def test_list_with_basic_types_is_simple(self):
        """Test that list[basic_type] is correctly identified as a simple type."""
        test_cases = [list[int], list[str], list[float], list[bool]]

        for response_model in test_cases:
            with self.subTest(response_model=response_model):
                self.assertTrue(
                    is_simple_type(response_model),
                    f"{response_model} should be a simple type",
                )

    def test_list_with_union_types_is_simple(self):
        """Test that list[Union[...]] is correctly identified as a simple type."""
        response_model = list[Union[int, str]]
        self.assertTrue(
            is_simple_type(response_model),
            "List[Union[int, str]] should be a simple type",
        )

    def test_list_with_pipe_union_is_simple(self):
        """Test that list[int | str] is correctly identified as a simple type."""
        if sys.version_info < (3, 10):
            self.skipTest("Union pipe syntax is only available in Python 3.10+")

        response_model = list[int | str]
        self.assertTrue(
            is_simple_type(response_model), "list[int | str] should be a simple type"
        )

    def test_empty_list_is_simple(self):
        """Test that plain list is correctly identified as a simple type."""
        response_model = list
        self.assertTrue(
            is_simple_type(response_model), "plain list should be a simple type"
        )


if __name__ == "__main__":
    unittest.main()
