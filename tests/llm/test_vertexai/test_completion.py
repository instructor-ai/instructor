import pytest
from vertexai.generative_models import GenerativeModel, GenerationResponse
from pydantic import BaseModel
import instructor


class User(BaseModel):
    name: str
    age: int


def test_create_with_completion():
    client = instructor.from_vertexai(
        client=GenerativeModel("gemini-1.5-pro-preview-0409"),
        mode=instructor.Mode.VERTEXAI_TOOLS,
    )

    resp, completion = client.chat.completions.create_with_completion(
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=User,
    )  # type: ignore

    assert isinstance(resp, User)
    assert resp.name == "Jason"
    assert resp.age == 25

    assert isinstance(completion, GenerationResponse)


def test_create_with_completion_bool():
    client = instructor.from_vertexai(
        client=GenerativeModel("gemini-1.5-pro-preview-0409"),
        mode=instructor.Mode.VERTEXAI_TOOLS,
    )

    resp, completion = client.chat.completions.create_with_completion(
        messages=[
            {
                "role": "user",
                "content": "Is Paris the capital of France?",
            }
        ],
        response_model=bool,  # type: ignore
    )

    assert isinstance(resp, bool)
    assert resp is True

    assert isinstance(completion, GenerationResponse)


@pytest.mark.asyncio
async def test_create_with_completion_async():
    client = instructor.from_vertexai(
        client=GenerativeModel("gemini-1.5-pro-preview-0409"),
        mode=instructor.Mode.VERTEXAI_TOOLS,
        _async=True,
    )

    resp = await client.chat.completions.create(  # type: ignore
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert resp.name == "Jason"
    assert resp.age == 25

    # assert isinstance(completion, GenerationResponse)
