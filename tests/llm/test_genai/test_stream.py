from itertools import product
from pydantic import BaseModel
import pytest
import instructor
from instructor.dsl.partial import Partial

from .util import models, modes


class UserExtract(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_partial_model(model, mode, client):
    client = instructor.from_genai(client, mode=mode)
    model = client.chat.completions.create(
        model=model,
        response_model=Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "{{ name }} is {{ age }} years old"},
        ],
        context={"name": "Jason", "age": 12},
    )
    final_model = None
    for m in model:  # type: ignore
        assert isinstance(m, UserExtract)
        final_model = m
    assert isinstance(final_model, UserExtract)
    assert final_model.age == 12
    assert final_model.name == "Jason"
