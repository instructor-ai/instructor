from typing import Any
from typing_extensions import TypedDict
from unittest.mock import AsyncMock, patch
from pydantic import BaseModel
import pytest
from instructor.process_response import handle_response_model, process_response_async


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


@pytest.mark.asyncio
async def test_process_response_async_calls_validate_async():

    class TestModel(BaseModel):
        async def validate_async(self) -> None:
            pass

        @classmethod
        def from_response(cls, response: Any) -> "TestModel":
            return cls()

    mock_response = AsyncMock()

    with patch.object(
        TestModel, "validate_async", new_callable=AsyncMock
    ) as mock_validate_async:
        with patch.object(TestModel, "from_response", return_value=TestModel()):
            await process_response_async(
                mock_response, response_model=TestModel
            )
        mock_validate_async.assert_called_once_with()
