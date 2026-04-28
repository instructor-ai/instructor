from unittest.mock import patch

import pytest

from instructor.core.exceptions import ConfigurationError
from instructor.providers.gemini.utils import map_to_gemini_function_schema


def test_map_to_gemini_function_schema_reports_missing_jsonref_dependency():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    original_import = __import__

    def mocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jsonref":
            raise ModuleNotFoundError("No module named 'jsonref'")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=mocked_import):
        with pytest.raises(ConfigurationError, match='instructor\\[google-genai\\]'):
            map_to_gemini_function_schema(schema)
