import sys

import pytest


def test_map_to_gemini_function_schema_requires_jsonref(monkeypatch):
    """ConfigurationError with install hint when jsonref is absent (Gemini schema path)."""
    from instructor.v2.core.errors import ConfigurationError
    from instructor.v2.providers.gemini.utils import map_to_gemini_function_schema

    monkeypatch.setitem(sys.modules, "jsonref", None)

    with pytest.raises(ConfigurationError) as excinfo:
        map_to_gemini_function_schema({"type": "object", "properties": {}})

    msg = str(excinfo.value)
    assert "instructor[google-genai]" in msg
    assert "uv pip install" in msg
