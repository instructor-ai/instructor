import pytest
import instructor
from instructor.function_calls import OpenAISchema, Mode
from openai import OpenAI


client = OpenAI()


class UserExtract(OpenAISchema):
    name: str
    age: int


def test_tool_call():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "user",
                "content": "Extract jason is 25 years old, mary is 30 years old",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": UserExtract.openai_schema,
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": UserExtract.openai_schema["name"]},
        },
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "UserExtract"
    assert tool_calls[0].function
    user = UserExtract.from_response(response, mode=Mode.TOOLS)
    assert user.name.lower() == "jason"
    assert user.age == 25


def test_json_mode():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"Make sure that your response to any message matchs the json_schema below, do not deviate at all: \n{UserExtract.model_json_schema()['properties']}",
            },
            {
                "role": "user",
                "content": "Extract jason is 25 years old",
            },
        ],
    )
    user = UserExtract.from_response(response, mode=Mode.JSON)
    assert user.name.lower() == "jason"
    assert user.age == 25


def test_markdown_json_mode():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"Make sure that your response to any message matchs the json_schema below, do not deviate at all: \n{UserExtract.model_json_schema()['properties']}",
            },
            {
                "role": "user",
                "content": "Extract jason is 25 years old",
            },
        ],
    )
    user = UserExtract.from_response(response, mode=Mode.MD_JSON)
    assert user.name.lower() == "jason"
    assert user.age == 25


@pytest.mark.parametrize("mode", [Mode.FUNCTIONS, Mode.JSON, Mode.TOOLS])
def test_mode(mode):
    client = instructor.patch(OpenAI(), mode=mode)
    user = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_model=UserExtract,
        messages=[
            {
                "role": "user",
                "content": "Extract jason is 25 years old",
            },
        ],
    )
    assert user.name.lower() == "jason"
    assert user.age == 25
