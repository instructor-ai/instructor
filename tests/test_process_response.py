from typing_extensions import TypedDict
from pydantic import BaseModel
from instructor.process_response import handle_response_model


def test_typed_dict_conversion() -> None:
    class User(TypedDict):  # type: ignore
        name: str
        age: int

    _, user_tool_definition = handle_response_model(User)

    class User(BaseModel):
        name: str
        age: int

    _, pydantic_user_tool_definition = handle_response_model(User)
    assert user_tool_definition == pydantic_user_tool_definition
