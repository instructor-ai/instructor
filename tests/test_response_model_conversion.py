from instructor.process_response import handle_response_model
from pydantic import BaseModel, Field
import instructor
import pytest

modes = [
    instructor.Mode.ANTHROPIC_JSON,
    instructor.Mode.JSON,
    instructor.Mode.MD_JSON,
    instructor.Mode.GEMINI_JSON,
    instructor.Mode.VERTEXAI_JSON,
]


def get_system_prompt(user_tool_definition, mode):
    if mode == instructor.Mode.ANTHROPIC_JSON:
        return user_tool_definition["system"]
    elif mode == instructor.Mode.GEMINI_JSON:
        return "\n".join(user_tool_definition["contents"][0]["parts"])
    elif mode == instructor.Mode.VERTEXAI_JSON:
        return str(user_tool_definition["generation_config"])
    return user_tool_definition["messages"][0]["content"]


@pytest.mark.parametrize("mode", modes)
def test_json_preserves_description_of_non_english_characters_in_json_mode(
    mode,
) -> None:
    messages = [
        {
            "role": "user",
            "content": "Extract the user from the text : 张三 20岁",
        }
    ]

    class User(BaseModel):
        name: str = Field(description="用户的名字")
        age: int = Field(description="用户的年龄")

    _, user_tool_definition = handle_response_model(User, mode=mode, messages=messages)

    system_prompt = get_system_prompt(user_tool_definition, mode)
    assert "用户的名字" in system_prompt
    assert "用户的年龄" in system_prompt

    _, user_tool_definition = handle_response_model(
        User,
        mode=mode,
        system="你是一个AI助手",
        messages=messages,
    )
    system_prompt = get_system_prompt(user_tool_definition, mode)
    assert "用户的名字" in system_prompt
    assert "用户的年龄" in system_prompt
