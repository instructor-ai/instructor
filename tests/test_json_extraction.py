"""
Tests for the improved JSON extraction functionality.
"""

import json
import pytest

from instructor.utils import extract_json_from_codeblock, extract_json_from_stream
from instructor.function_calls import _extract_text_content, _validate_model_from_json
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    skills: list[str] = []


class TestJSONExtraction:
    """Test the improved JSON extraction functionality."""

    def test_extract_from_codeblock(self):
        """Test extracting JSON from markdown code blocks."""
        # JSON inside markdown code block
        markdown_json = """
        # Test Data
        Here is some data:
        ```json
        {
          "name": "John",
          "age": 30,
          "skills": ["python", "javascript"]
        }
        ```
        More text here.
        """

        result = extract_json_from_codeblock(markdown_json)
        parsed = json.loads(result)

        assert parsed["name"] == "John"
        assert parsed["age"] == 30
        assert "python" in parsed["skills"]

    def test_extract_from_codeblock_no_language(self):
        """Test extracting JSON from code blocks without language specified."""
        # JSON inside unmarked code block
        markdown_json = """
        # Test Data
        Here is some data:
        ```
        {
          "name": "Jane",
          "age": 25,
          "skills": ["java", "typescript"]
        }
        ```
        More text here.
        """

        result = extract_json_from_codeblock(markdown_json)
        parsed = json.loads(result)

        assert parsed["name"] == "Jane"
        assert parsed["age"] == 25
        assert "java" in parsed["skills"]

    def test_extract_plain_json(self):
        """Test extracting JSON without code blocks."""
        # Plain JSON with surrounding text
        plain_json = """
        Here is the user information:
        {
          "name": "Bob",
          "age": 40,
          "skills": ["go", "rust"]
        }
        End of data.
        """

        result = extract_json_from_codeblock(plain_json)
        parsed = json.loads(result)

        assert parsed["name"] == "Bob"
        assert parsed["age"] == 40
        assert "rust" in parsed["skills"]

    def test_nested_json(self):
        """Test extracting nested JSON objects."""
        # Nested JSON
        nested_json = """
        ```json
        {
          "name": "Alice",
          "age": 35,
          "address": {
            "street": "123 Main St",
            "city": "Anytown",
            "zip": "12345"
          },
          "skills": ["python", "ml"]
        }
        ```
        """

        result = extract_json_from_codeblock(nested_json)
        parsed = json.loads(result)

        assert parsed["name"] == "Alice"
        assert parsed["address"]["city"] == "Anytown"
        assert "ml" in parsed["skills"]

    def test_json_with_arrays(self):
        """Test extracting JSON with arrays."""
        # JSON with arrays
        array_json = """
        Here's an array of users:
        {
          "users": [
            {"name": "User1", "age": 20},
            {"name": "User2", "age": 30},
            {"name": "User3", "age": 40}
          ],
          "total": 3
        }
        """

        result = extract_json_from_codeblock(array_json)
        parsed = json.loads(result)

        assert len(parsed["users"]) == 3
        assert parsed["users"][0]["name"] == "User1"
        assert parsed["total"] == 3

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        # Invalid JSON
        invalid_json = """
        This is not valid JSON:
        { name: "Test" }
        """

        result = extract_json_from_codeblock(invalid_json)
        # Should return the content between braces even if it's invalid
        assert "{" in result and "}" in result

        # Should raise when trying to parse
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    def test_extract_from_stream(self):
        """Test extracting JSON from a stream of chunks."""
        # JSON split into chunks
        chunks = [
            '{"na',
            'me": "',
            "Stream",
            'User", ',
            '"age": 45, "sk',
            'ills": ["stream',
            'ing", "json"]}',
        ]

        collected = "".join(extract_json_from_stream(chunks))
        parsed = json.loads(collected)

        assert parsed["name"] == "StreamUser"
        assert parsed["age"] == 45
        assert "streaming" in parsed["skills"]


class TestTextExtraction:
    """Test the text extraction utilities."""

    def test_extract_text_openai_format(self):
        """Test extracting text from OpenAI completion format."""

        class MockMessage:
            content = "Sample content"

        class MockChoice:
            message = MockMessage()

        class MockCompletion:
            choices = [MockChoice()]

        completion = MockCompletion()
        result = _extract_text_content(completion)

        assert result == "Sample content"

    def test_extract_text_simple_format(self):
        """Test extracting text from simple text format."""

        class MockCompletion:
            text = "Simple text response"

        completion = MockCompletion()
        result = _extract_text_content(completion)

        assert result == "Simple text response"

    def test_extract_text_anthropic_format(self):
        """Test extracting text from Anthropic format."""

        class MockTextBlock:
            def __init__(self, text_content):
                self.type = "text"
                self.text = text_content

        class MockCompletion:
            content = [
                MockTextBlock("Anthropic response"),
                MockTextBlock("Additional text"),
            ]

        completion = MockCompletion()
        result = _extract_text_content(completion)

        assert result == "Anthropic response"

    def test_extract_text_bedrock_format(self):
        """Test extracting text from Bedrock format."""
        completion = {
            "output": {"message": {"content": [{"text": "Bedrock response"}]}}
        }

        result = _extract_text_content(completion)
        assert result == "Bedrock response"

    def test_extract_text_unknown_format(self):
        """Test extracting text from unknown format."""

        class UnknownFormat:
            unknown_field = "Can't extract this"

        completion = UnknownFormat()
        result = _extract_text_content(completion)

        # Should return empty string for unknown formats
        assert result == ""


class TestModelValidation:
    """Test the model validation utilities."""

    def test_validate_model_strict(self):
        """Test model validation with strict mode."""
        json_str = '{"name": "ValidUser", "age": 30, "skills": ["coding"]}'
        result = _validate_model_from_json(Person, json_str, None, True)

        assert result.name == "ValidUser"
        assert result.age == 30
        assert result.skills == ["coding"]

    def test_validate_model_non_strict(self):
        """Test model validation with non-strict mode."""
        # In non-strict mode, string numbers can be coerced to integers
        json_str = '{"name": "NonStrictUser", "age": "25", "skills": ["testing"]}'
        result = _validate_model_from_json(Person, json_str, None, False)

        assert result.name == "NonStrictUser"
        assert result.age == 25  # String "25" coerced to integer
        assert result.skills == ["testing"]

    def test_validate_model_json_error(self):
        """Test handling JSON decode errors."""
        invalid_json = '{"name": "Invalid, "age": 20}'  # Missing quote

        with pytest.raises(Exception) as excinfo:
            _validate_model_from_json(Person, invalid_json, None, True)

        # Pydantic directly raises validation errors now, not our custom message
        assert "Invalid JSON" in str(excinfo.value)
