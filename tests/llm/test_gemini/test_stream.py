from itertools import product
from collections.abc import Iterable
from pydantic import BaseModel
import pytest
import instructor
import google.generativeai as genai
from instructor.dsl.partial import Partial

from .util import models, modes


class UserExtract(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model, mode, stream", product(models, modes, [True, False]))
def test_iterable_model(model, mode, stream):
    client = instructor.from_gemini(genai.GenerativeModel(model), mode=mode)
    model = client.chat.completions.create(
        response_model=Iterable[UserExtract],
        max_retries=2,
        stream=stream,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    for m in model:
        assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_partial_model(model, mode):
    client = instructor.from_gemini(genai.GenerativeModel(model), mode=mode)
    model = client.chat.completions.create(
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "{{ name }} is {{ age }} years old"},
        ],
        context={"name": "Jason", "age": 12},
    )
    final_model = None
    for m in model:
        assert isinstance(m, UserExtract)
        final_model = m

    assert final_model.age == 12
    assert final_model.name == "Jason"
