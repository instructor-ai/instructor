import sys
import unittest
from unittest import mock

import instructor
from pydantic import BaseModel


@unittest.skipIf(sys.version_info < (3, 10), "Union pipe syntax requires Python 3.10+")
class TestPipeSyntax(unittest.TestCase):
    def test_partial_with_pipe_syntax(self):
        """Test that create_partial works with pipe syntax for Union types."""
        class UserWithPipe(BaseModel):
            name: str
            age: int
            category: str | None
        
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock.MagicMock()
        
        client = instructor.from_openai(mock_client)
        response_model = instructor.Partial[UserWithPipe]
        self.assertIsNotNone(response_model)
