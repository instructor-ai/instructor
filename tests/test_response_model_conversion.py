from instructor.process_response import handle_response_model
from pydantic import BaseModel, Field
import instructor


def test_anthropic_json_preserves_description_of_non_english_characters() -> None:
    class User(BaseModel):
        name: str = Field(description="用户的名字")
        age: int = Field(description="用户的年龄")

    _, user_tool_definition = handle_response_model(
        User, mode=instructor.Mode.ANTHROPIC_JSON
    )
    print(user_tool_definition["system"])
    assert "用户的名字" in user_tool_definition["system"]
    assert "用户的年龄" in user_tool_definition["system"]

    _, user_tool_definition = handle_response_model(
        User, mode=instructor.Mode.ANTHROPIC_JSON, system="你是一个AI助手"
    )
    print(user_tool_definition["system"])
    assert "用户的名字" in user_tool_definition["system"]
    assert "用户的年龄" in user_tool_definition["system"]
