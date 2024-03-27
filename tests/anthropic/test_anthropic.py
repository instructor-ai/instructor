from enum import Enum
from typing import List
from unittest.mock import patch

import anthropic
import pytest
from pydantic import BaseModel

import instructor

create = instructor.patch(create=anthropic.Anthropic().messages.create, mode=instructor.Mode.ANTHROPIC_TOOLS)


def _get_base_response_message():
    from anthropic import ContentBlock, Message

    return Message(id="", content=[ContentBlock()], model="", role="assistant")


# @pytest.mark.skip
def test_anthropic():
    class Properties(BaseModel):
        name: str
        value: List[str]

    class User(BaseModel):
        name: str
        age: int
        properties: List[Properties]

    with patch("anthropic.Anthropic.messages.create") as mock_create:
        mock_create.return_value = "<function_calls>\n<invoke>\n<tool_name>User</tool_name>\n<parameters>\n<name>Alice</name>\n<age>25</age>\n<properties>\n<name>gender</name>\n<value>female</value>\n</properties>\n<properties>\n<name>occupation</name>\n<value>engineer</value>\n</properties>\n<properties>\n<name>hobbies</name>\n<value>reading</value>\n<value>hiking</value>\n<value>traveling</value>\n</properties>\n</parameters>\n</invoke>\n</function_calls>"
        resp = create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            max_retries=0,
            messages=[
                {
                    "role": "user",
                    "content": "Create a user for a model with a name, age, and properties.",
                }
            ],
            response_model=User,
        )  # type: ignore

    assert isinstance(resp, User)


# @pytest.mark.skip
def test_anthropic_enum():
    class ProgrammingLanguage(Enum):
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        TYPESCRIPT = "typescript"
        UNKNOWN = "unknown"
        OTHER = "other"

    class SimpleEnum(BaseModel):
        language: ProgrammingLanguage

    with patch("anthropic.Anthropic.messages.create") as mock_create:
        mocked_message = _get_base_response_message()
        mocked_message.content[
            0
        ].text = "<function_calls>\n<invoke>\n<tool_name>SimpleEnum</tool_name>\n<parameters>\n<language>python</language>\n</parameters>\n</invoke>\n</function_calls>\n\nclass User:\n    def __init__(self, name, age, **properties):\n        self.name = name\n        self.age = age\n        self.properties = properties\n\n<function_calls>\n<invoke>\n<tool_name>SimpleEnum</tool_name>\n<parameters>\n<language>javascript</language>\n</parameters>\n</invoke>\n</function_calls>\n\nclass User {\n  constructor(name, age, properties = {}) {\n    this.name = name;\n    this.age = age;\n    this.properties = properties;\n  }\n}\n\n<function_calls>\n<invoke>\n<tool_name>SimpleEnum</tool_name>\n<parameters>\n<language>typescript</language>\n</parameters>\n</invoke>\n</function_calls>"
        mock_create.return_value = mocked_message
        resp = create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            max_retries=0,
            messages=[
                {
                    "role": "user",
                    "content": "What is your favorite programming language?",
                }
            ],
            response_model=SimpleEnum,
        )  # type: ignore

    assert isinstance(resp, SimpleEnum)
