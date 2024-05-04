from typing import TypeVar

import pytest
from anthropic.types import Message, Usage
from openai.resources.chat.completions import ChatCompletion
from pydantic import BaseModel, ValidationError

import instructor
from instructor import OpenAISchema, openai_schema
from instructor.exceptions import IncompleteOutputException

T = TypeVar("T")


@pytest.fixture  # type: ignore[misc]
def test_model() -> type[OpenAISchema]:
    class TestModel(OpenAISchema):  # type: ignore[misc]
        name: str = "TestModel"
        data: str

    return TestModel


@pytest.fixture  # type: ignore[misc]
def mock_completion(request: T) -> ChatCompletion:
    finish_reason = "stop"
    data_content = '{\n"data": "complete data"\n}'

    if hasattr(request, "param"):
        finish_reason = request.param.get("finish_reason", finish_reason)
        data_content = request.param.get("data_content", data_content)

    mock_choices = [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "function_call": {"name": "TestModel", "arguments": data_content},
                "content": data_content,
            },
            "finish_reason": finish_reason,
        }
    ]

    completion = ChatCompletion(
        id="test_id",
        choices=mock_choices,
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion",
    )

    return completion

@pytest.fixture  # type: ignore[misc]
def mock_anthropic_message(request: T) -> Message:
    data_content = '{\n"data": "Claude says hi"\n}'
    if hasattr(request, "param"):
        data_content = request.param.get("data_content", data_content)
    return Message(
        id="test_id",
        content=[{ "type": "text", "text": data_content }],
        model="claude-3-haiku-20240307",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=100,
            output_tokens=100,
        )
    )

def test_openai_schema() -> None:
    @openai_schema
    class Dataframe(BaseModel):  # type: ignore[misc]
        """
        Class representing a dataframe. This class is used to convert
        data into a frame that can be used by pandas.
        """

        data: str
        columns: str

        def to_pandas(self) -> None:
            pass

    assert hasattr(Dataframe, "openai_schema")
    assert hasattr(Dataframe, "from_response")
    assert hasattr(Dataframe, "to_pandas")
    assert Dataframe.openai_schema["name"] == "Dataframe"


def test_openai_schema_raises_error() -> None:
    with pytest.raises(TypeError, match="must be a subclass of pydantic.BaseModel"):

        @openai_schema
        class Dummy:
            pass


def test_no_docstring() -> None:
    class Dummy(OpenAISchema):  # type: ignore[misc]
        attr: str

    assert (
        Dummy.openai_schema["description"]
        == "Correctly extracted `Dummy` with all the required parameters with correct types"
    )


@pytest.mark.parametrize(
    "mock_completion",
    [{"finish_reason": "length", "data_content": '{\n"data": "incomplete dat"\n}'}],
    indirect=True,
)  # type: ignore[misc]
def test_incomplete_output_exception(
    test_model: type[OpenAISchema], mock_completion: ChatCompletion
) -> None:
    with pytest.raises(IncompleteOutputException):
        test_model.from_response(mock_completion, mode=instructor.Mode.FUNCTIONS)


def test_complete_output_no_exception(
    test_model: type[OpenAISchema], mock_completion: ChatCompletion
) -> None:
    test_model_instance = test_model.from_response(
        mock_completion, mode=instructor.Mode.FUNCTIONS
    )
    assert test_model_instance.data == "complete data"


@pytest.mark.asyncio  # type: ignore[misc]
@pytest.mark.parametrize(
    "mock_completion",
    [{"finish_reason": "length", "data_content": '{\n"data": "incomplete dat"\n}'}],
    indirect=True,
)  # type: ignore[misc]
def test_incomplete_output_exception_raise(
    test_model: type[OpenAISchema], mock_completion: ChatCompletion
) -> None:
    with pytest.raises(IncompleteOutputException):
        test_model.from_response(mock_completion, mode=instructor.Mode.FUNCTIONS)

def test_anthropic_no_exception(
    test_model: type[OpenAISchema], mock_anthropic_message: Message
) -> None:
    test_model_instance = test_model.from_response(mock_anthropic_message, mode=instructor.Mode.ANTHROPIC_JSON)
    assert test_model_instance.data == "Claude says hi"

@pytest.mark.parametrize(
    "mock_anthropic_message",
    [{"data_content": '{\n"data": "Claude likes\ncontrol\ncharacters"\n}'}],
    indirect=True,
)  # type: ignore[misc]
def test_control_characters_not_allowed_in_anthropic_json_strict_mode(
    test_model: type[OpenAISchema], mock_anthropic_message: Message
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        test_model.from_response(
            mock_anthropic_message, mode=instructor.Mode.ANTHROPIC_JSON, strict=True
        )

    # https://docs.pydantic.dev/latest/errors/validation_errors/#json_invalid
    exc = exc_info.value
    assert len(exc.errors()) == 1
    assert exc.errors()[0]["type"] == "json_invalid"
    assert "control character" in exc.errors()[0]["msg"]

@pytest.mark.parametrize(
    "mock_anthropic_message",
    [{"data_content": '{\n"data": "Claude likes\ncontrol\ncharacters"\n}'}],
    indirect=True,
)  # type: ignore[misc]
def test_control_characters_allowed_in_anthropic_json_non_strict_mode(
    test_model: type[OpenAISchema], mock_anthropic_message: Message
) -> None:
    test_model_instance = test_model.from_response(
        mock_anthropic_message, mode=instructor.Mode.ANTHROPIC_JSON, strict=False
    )
    assert test_model_instance.data == "Claude likes\ncontrol\ncharacters"
