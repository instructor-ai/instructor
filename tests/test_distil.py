
import unittest
from typing import Generator
import sys
from pydantic import BaseModel
from instructor.distil import (
    get_signature_from_fn,
    format_function,
    is_return_type_base_model_or_instance,
    Instructions,
)
from unittest.mock import MagicMock, patch

class User(BaseModel):
    name: str
    age: int

def sample_function(a: int, b: int) -> User:
    """Sample docstring."""
    return User(name="Test", age=a + b)

class TestDistil(unittest.TestCase):
    def setUp(self):
        # Patch OpenAI to avoid missing API key errors
        self.openai_patcher = patch("instructor.distil.OpenAI")
        self.mock_openai = self.openai_patcher.start()
        
        # Patch instructor.distil.patch to avoid actual patching logic
        self.instructor_patch_patcher = patch("instructor.distil.patch")
        self.mock_instructor_patch = self.instructor_patch_patcher.start()
        self.mock_instructor_patch.side_effect = lambda c, mode=None: c

    def tearDown(self):
        self.openai_patcher.stop()
        self.instructor_patch_patcher.stop()

    def test_get_signature_from_fn(self):
        sig = get_signature_from_fn(sample_function)
        self.assertIn("def sample_function(a: int, b: int) -> ", sig)
        self.assertIn("Sample docstring", sig)

    def test_format_function(self):
        formatted = format_function(sample_function)
        self.assertIn("def sample_function", formatted)
        self.assertIn("return User(name=\"Test\", age=a + b)", formatted)
        self.assertIn("Sample docstring", formatted)

    def test_is_return_type_base_model_or_instance(self):
        self.assertTrue(is_return_type_base_model_or_instance(sample_function))
        
        def invalid_function():
            pass
            
        def int_function() -> int:
            return 1

        with self.assertRaises(AssertionError):
            is_return_type_base_model_or_instance(invalid_function)
            
        self.assertFalse(is_return_type_base_model_or_instance(int_function))

    def test_instructions_init_patches_client(self):
        # Test that client is patched
        Instructions()
        self.mock_instructor_patch.assert_called()

    def test_instructions_distil_decorator(self):
        instructions = Instructions()
        # Mock the client
        instructions.client = MagicMock()
        instructions.track = MagicMock()
        
        @instructions.distil
        def tracked_function(x: int) -> User:
            return User(name="Tracked", age=x)
            
        result = tracked_function(10)
        self.assertEqual(result.name, "Tracked")
        self.assertEqual(result.age, 10)
        instructions.track.assert_called_once()
        
    def test_instructions_dispatch_mode(self):
        instructions = Instructions()
        # Mock the client
        instructions.client = MagicMock()
        instructions.client.chat.completions.create.return_value = User(name="Dispatched", age=20)
        
        @instructions.distil(mode="dispatch")
        def dispatched_function(x: int) -> User:
            return User(name="Original", age=x)
            
        result = dispatched_function(20)
        
        # In dispatch mode, it should call client.create and return its result
        self.assertEqual(result.name, "Dispatched")
        instructions.client.chat.completions.create.assert_called_once()
        
        # Verify arguments passed to create
        call_kwargs = instructions.client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["response_model"], User)
        self.assertIn("messages", call_kwargs)

if __name__ == "__main__":
    unittest.main()
