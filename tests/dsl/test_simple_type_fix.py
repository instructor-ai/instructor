import sys
import unittest
from typing import Union, List  # noqa: UP035
from typing import get_origin, get_args
from instructor.dsl.simple_type import is_simple_type


class TestSimpleTypeFix(unittest.TestCase):
    def test_list_with_union_type(self):
        """Test that list[int | str] is correctly identified as a simple type."""
        # This is the type that was failing in Python 3.10
        if sys.version_info < (3, 10):
            self.skipTest("Union pipe syntax is only available in Python 3.10+")
        response_model = list[int | str]
        self.assertTrue(
            is_simple_type(response_model),
            f"list[int | str] should be a simple type in Python {sys.version_info.major}.{sys.version_info.minor}. Instead it was identified as {type(response_model)} with origin {get_origin(response_model)} and args {get_args(response_model)}",
        )

    def test_list_with_union_type_alternative_syntax(self):
        """Test that List[Union[int, str]] is correctly identified as a simple type."""
        # Alternative syntax
        response_model = List[Union[int, str]]  # noqa: UP006
        self.assertTrue(
            is_simple_type(response_model),
            f"List[Union[int, str]] should be a simple type in Python {sys.version_info.major}.{sys.version_info.minor}",
        )

