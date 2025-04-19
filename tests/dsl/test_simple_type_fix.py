import sys
import unittest
from typing import Union

from instructor.dsl.simple_type import is_simple_type


class TestSimpleTypeFix(unittest.TestCase):
    def test_list_with_union_type(self):
        """Test that list with union types is correctly identified as a simple type."""
        # Use different syntax based on Python version
        if sys.version_info >= (3, 10):
            # Python 3.10+ supports pipe syntax
            response_model = list[int | str]  # type: ignore
            self.assertTrue(
                is_simple_type(response_model),
                f"list[int | str] should be a simple type in Python {sys.version_info.major}.{sys.version_info.minor}",
            )
        else:
            # For Python 3.9, test the same functionality but with Union
            response_model = list[Union[int, str]]
            self.assertTrue(
                is_simple_type(response_model),
                f"list[Union[int, str]] should be a simple type in Python {sys.version_info.major}.{sys.version_info.minor}",
            )

    def test_list_with_union_type_alternative_syntax(self):
        """Test that list[Union[int, str]] is correctly identified as a simple type."""
        # Using Union syntax that works in all Python versions
        response_model = list[Union[int, str]]
        self.assertTrue(
            is_simple_type(response_model),
            f"list[Union[int, str]] should be a simple type in Python {sys.version_info.major}.{sys.version_info.minor}",
        )


if __name__ == "__main__":
    unittest.main()
