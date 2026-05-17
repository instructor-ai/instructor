from pydantic import BaseModel

from openai import pydantic_function_tool

from instructor.v2.providers.openai.handlers import (
    OpenAIResponsesToolsHandler,
)


class ResponseToolModel(BaseModel):
    """Extract a structured response for the user."""

    name: str


def test_responses_tools_preserves_function_description() -> None:
    expected_description = pydantic_function_tool(ResponseToolModel)["function"][
        "description"
    ]

    _, responses_tools_kwargs = OpenAIResponsesToolsHandler().prepare_request(
        ResponseToolModel, {}
    )
    _, inbuilt_tools_kwargs = OpenAIResponsesToolsHandler().prepare_request(
        ResponseToolModel, {}
    )

    assert responses_tools_kwargs["tools"][0]["description"] == expected_description
    assert inbuilt_tools_kwargs["tools"][0]["description"] == expected_description
    assert responses_tools_kwargs["tools"][0] == inbuilt_tools_kwargs["tools"][0]
