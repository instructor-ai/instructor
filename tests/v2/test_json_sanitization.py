from __future__ import annotations

import pytest
from pydantic import BaseModel

from instructor.v2.core.function_calls import _validate_model_from_json

class DummyModel(BaseModel):
    name: str

def test_validate_model_from_json_sanitizes_markdown():
    # Messy string wrapped in markdown and whitespace
    messy_json = "   \n```json\n{\"name\": \"Alice\"}\n```\n   "
    
    # Should correctly sanitize and parse
    model = _validate_model_from_json(DummyModel, messy_json)
    assert model.name == "Alice"

def test_validate_model_from_json_empty_string_raises_value_error():
    with pytest.raises(ValueError, match="Empty response: Cannot parse JSON from an empty string"):
        _validate_model_from_json(DummyModel, "   \n  ")
