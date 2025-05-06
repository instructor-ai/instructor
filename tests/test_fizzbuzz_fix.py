import unittest
import sys
from instructor.dsl.simple_type import is_simple_type
from instructor.process_response import prepare_response_model


class TestFizzbuzzFix(unittest.TestCase):
    def test_fizzbuzz_response_model(self):
        if sys.version_info < (3, 10):
            self.skipTest("Union pipe syntax is only available in Python 3.10+")
        """Test that list[int | str] works correctly as a response model."""
        # This is the type used in the fizzbuzz example
        response_model = list[int | str]

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
