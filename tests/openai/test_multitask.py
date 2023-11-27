
from typing import Iterable
from openai import OpenAI
from pydantic import BaseModel

import instructor
from instructor.function_calls import OpenAISchema, Mode


class User(BaseModel):
    name: str
    age: int

Users = Iterable[User]



def test_multi_user_function_mode():
    client = instructor.patch(OpenAI(), mode=Mode.FUNCTIONS)
    def stream_extract(input: str) -> Iterable[User]:
        return client.chat.completions.create(
            model="gpt-3.5-turbo",
            stream=True,
            response_model=Users,
            messages=[
                {
                    "role": "system",
                    "content": "You are a perfect entity extraction system",
                },
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}"
                        "Correctly segment it into entitites"
                        "Make sure the JSON is correct"
                    ),
                },
            ],
            max_tokens=1000,
        )

    resp = [user for user in stream_extract(input="Jason is 20, Sarah is 30")]
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30

def test_multi_user_json_mode():
    client = instructor.patch(OpenAI(), mode=Mode.JSON)
    def stream_extract(input: str) -> Iterable[User]:
        return client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            stream=True,
            response_model=Users,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}"
                        "Correctly segment it into entitites"
                        "Make sure the JSON is correct"
                    ),
                },
            ],
            max_tokens=1000,
        )

    resp = [user for user in stream_extract(input="Jason is 20, Sarah is 30")]
    print(resp)
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30

def test_multi_user_tools_mode():
    client = instructor.patch(OpenAI(), mode=Mode.TOOLS)
    def stream_extract(input: str) -> Iterable[User]:
        return client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            stream=True,
            response_model=Users,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}"
                        "Correctly segment it into entitites"
                        "Make sure the JSON is correct"
                    ),
                },
            ],
            max_tokens=1000,
        )

    resp = [user for user in stream_extract(input="Jason is 20, Sarah is 30")]
    print(resp)
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30

def test_multi_user_legacy():
    def stream_extract(input: str, cls) -> Iterable[User]:
        client = instructor.patch(OpenAI())
        MultiUser = instructor.MultiTask(cls)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            stream=True,
            functions=[MultiUser.openai_schema],
            function_call={"name": MultiUser.openai_schema["name"]},
            messages=[
                {
                    "role": "system",
                    "content": "You are a perfect entity extraction system",
                },
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}"
                        "Correctly segment it into entitites"
                        "Make sure the JSON is correct"
                    ),
                },
            ],
            max_tokens=1000,
        )
        return MultiUser.from_streaming_response(completion, mode=Mode.FUNCTIONS)

    resp = [user for user in stream_extract(input="Jason is 20, Sarah is 30", cls=User)]
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30
