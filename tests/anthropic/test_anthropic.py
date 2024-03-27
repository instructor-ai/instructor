import os
from enum import Enum
from typing import List
from unittest.mock import patch

import anthropic
import pytest
from pydantic import BaseModel

import instructor


def _get_base_response_message():
    from anthropic.types import ContentBlock, Message, Usage

    return Message(
        id="",
        content=[ContentBlock(text="", type="text")],
        model="",
        role="assistant",
        type="message",
        usage=Usage(input_tokens=0, output_tokens=0),
    )


def test_anthropic():
    class Property(BaseModel):
        name: str
        value: str

    class User(BaseModel):
        name: str
        age: int
        properties: List[Property]

    with patch("anthropic.resources.messages.Messages.create") as mock_create:
        message = _get_base_response_message()
        message.content[
            0
        ].text = "<function_calls><invoke><tool_name>User</tool_name><parameters><name>John Smith</name><age>35</age><properties><name>hair_color</name><value>brown</value></properties><properties><name>eye_color</name><value>blue</value></properties><properties><name>occupation</name><value>software engineer</value></properties></parameters></invoke></function_calls>"
        create = instructor.patch(create=anthropic.Anthropic().messages.create, mode=instructor.Mode.ANTHROPIC_TOOLS)
        mock_create.return_value = message
        resp = create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            max_retries=0,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": "Create a user for a model with a name, age, and properties.",
                }
            ],
            response_model=User,
        )  # type: ignore

    assert isinstance(resp, User)


def test_anthropic_enum():
    class ProgrammingLanguage(Enum):
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        TYPESCRIPT = "typescript"
        UNKNOWN = "unknown"
        OTHER = "other"

    class SimpleEnum(BaseModel):
        language: ProgrammingLanguage

    with patch("anthropic.resources.messages.Messages.create") as mock_create:
        message = _get_base_response_message()
        message.content[
            0
        ].text = "<function_calls>\n<invoke>\n<tool_name>SimpleEnum</tool_name>\n<parameters>\n<language>python</language>\n</parameters>\n</invoke>\n</function_calls>"
        mock_create.return_value = message
        create = instructor.patch(create=anthropic.Anthropic().messages.create, mode=instructor.Mode.ANTHROPIC_TOOLS)
        resp = create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            max_retries=0,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": "What is your favorite programming language?",
                }
            ],
            response_model=SimpleEnum,
        )  # type: ignore

    assert isinstance(resp, SimpleEnum)


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    pytest.main([current_file])
