import pytest
from pydantic import BaseModel
import instructor
from .util import models, modes


class User(BaseModel):
    name: str
    age: int


class Users(BaseModel):
    users: list[User]


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_simple_string_message(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=["Ivan is 28 years old"],  # type: ignore
        response_model=Users,
    )
    assert isinstance(response, Users)
    assert len(response.users) > 0
    assert response.users[0].name == "Ivan"
    assert response.users[0].age == 28


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_system_prompt(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Ivan is 28 years old",
            },
            {
                "role": "user",
                "content": "Make sure that the response is a list of users",
            },
        ],
        response_model=Users,
    )
    assert isinstance(response, Users)
    assert len(response.users) > 0
    assert response.users[0].name == "Ivan"
    assert response.users[0].age == 28


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_system_kwarg(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        system="Ivan is 28 years old",
        messages=[
            {
                "role": "user",
                "content": "Make sure that the response is a list of users",
            },
        ],
        response_model=Users,
    )
    assert isinstance(response, Users)
    assert len(response.users) > 0
    assert response.users[0].name == "Ivan"
    assert response.users[0].age == 28


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_system_prompt_list(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    "Ivan is",
                    " 28 years old",
                ],
            },  # type: ignore
            {
                "role": "user",
                "content": "Make sure that the response is a list of users",
            },
        ],
        response_model=Users,
    )
    assert isinstance(response, Users)
    assert len(response.users) > 0
    assert response.users[0].name == "Ivan"
    assert response.users[0].age == 28
