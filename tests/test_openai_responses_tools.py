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


def test_responses_tools_aligns_text_format() -> None:
    from instructor.v2.providers.openai.handlers import OpenAIResponsesToolsHandler
    _, kwargs = OpenAIResponsesToolsHandler().prepare_request(ResponseToolModel, {})
    assert "text" in kwargs
    assert kwargs["text"]["format"] == "json_schema"
    assert kwargs["text"]["json_schema"]["name"] == "ResponseToolModel"
    assert "schema" in kwargs["text"]["json_schema"]
    assert kwargs["text"]["json_schema"]["strict"] is True


def test_reask_responses_tools_targeted_message() -> None:
    from instructor.v2.providers.openai.handlers import reask_responses_tools
    class MockToolCall:
        def __init__(self, arguments, name=None):
            self.arguments = arguments
            self.name = name
            self.type = "tool_call"
    
    class MockResponse:
        def __init__(self, output):
            self.output = output

    response = MockResponse(output=[MockToolCall(arguments="{}")])
    kwargs = {"messages": []}
    new_kwargs = reask_responses_tools(kwargs, response, ValueError("Some validation error"))
    assert "Tool called with empty arguments" in new_kwargs["messages"][0]["content"]

    response = MockResponse(output=[MockToolCall(arguments='{"name": "test"}')])
    kwargs = {"messages": []}
    new_kwargs = reask_responses_tools(kwargs, response, ValueError("Some validation error"))
    assert "Recall the function correctly, fix the errors with" in new_kwargs["messages"][0]["content"]
