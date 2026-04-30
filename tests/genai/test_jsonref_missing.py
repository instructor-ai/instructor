"""Tests for the helpful ImportError raised when jsonref is missing.

See https://github.com/567-labs/instructor/issues/2288.
"""

from __future__ import annotations

import sys

import pytest


def test_map_to_gemini_function_schema_raises_helpful_error_without_jsonref(monkeypatch):
    """If `jsonref` cannot be imported, the user should see an actionable
    message pointing to the `[google-genai]` extra rather than a bare
    ModuleNotFoundError surfacing deep in the call stack.
    """
    # Force `import jsonref` inside map_to_gemini_function_schema to raise.
    # monkeypatch.setitem restores the original sys.modules entry after the
    # test, so this doesn't leak across the suite.
    monkeypatch.setitem(sys.modules, "jsonref", None)

    from instructor.providers.gemini.utils import map_to_gemini_function_schema

    with pytest.raises(ImportError) as excinfo:
        map_to_gemini_function_schema({"type": "object", "properties": {}})

    msg = str(excinfo.value)
    assert "jsonref" in msg
    assert "instructor[google-genai]" in msg
