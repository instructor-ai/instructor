import unittest
import sys
from typing import Union
from instructor.dsl.simple_type import is_simple_type
from instructor.process_response import prepare_response_model


class TestFizzbuzzFix(unittest.TestCase):
    def test_fizzbuzz_response_model(self):
        """Test that list with union types works correctly as a response model."""
        # Use different syntax based on Python version
        if sys.version_info >= (3, 10):
            # Python 3.10+ supports pipe syntax
            response_model = list[int | str]  # type: ignore

            # First check that it's correctly identified as a simple type
            self.assertTrue(
                is_simple_type(response_model),
                f"list[int | str] should be a simple type in Python {sys.version_info.major}.{sys.version_info.minor}",
            )

            # Then check that prepare_response_model handles it correctly
            prepared_model = prepare_response_model(response_model)
            self.assertIsNotNone(
                prepared_model,
                "prepare_response_model should not return None for list[int | str]",
            )
        else:
            # For Python 3.9, use Union syntax instead
            response_model = list[Union[int, str]]

            # First check that it's correctly identified as a simple type
            self.assertTrue(
                is_simple_type(response_model),
                f"list[Union[int, str]] should be a simple type in Python {sys.version_info.major}.{sys.version_info.minor}",
            )

            # Then check that prepare_response_model handles it correctly
            prepared_model = prepare_response_model(response_model)
            self.assertIsNotNone(
                prepared_model,
                "prepare_response_model should not return None for list[Union[int, str]]",
            )


if __name__ == "__main__":
    unittest.main()
