from collections.abc import Iterable
from pydantic import BaseModel
import instructor
import vertexai.generative_models as gm #type: ignore[reportMissingTypeStubs]
from instructor.dsl.partial import Partial
from .util import model, mode


class UserExtract(BaseModel):
    name: str
    age: int


def test_iterable_model():
    client = instructor.from_vertexai(gm.GenerativeModel(model), mode=mode)
    response_stream = client.chat.completions.create(
        response_model=Iterable[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    for chunk in response_stream:
        assert isinstance(chunk, UserExtract)


def test_partial_model():
    client = instructor.from_vertexai(gm.GenerativeModel(model), mode=mode)
    response_stream = client.chat.completions.create(
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Anibal is 23 years old"},
        ],
    )
    for chunk in response_stream:
        assert isinstance(chunk, UserExtract)
