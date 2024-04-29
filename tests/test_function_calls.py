from typing import TypeVar
import pytest
from pydantic import BaseModel
from openai.resources.chat.completions import ChatCompletion

from instructor import openai_schema, OpenAISchema
import instructor
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
