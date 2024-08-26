import instructor
import pytest
import enum
import vertexai.generative_models as gm  # type: ignore
from itertools import product
from typing import Literal

from .util import models, modes


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_literal(model, mode):
    client = instructor.from_vertexai(gm.GenerativeModel(model), mode)

    response = client.create(
        response_model=Literal["1231", "212", "331"],
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert response in ["1231", "212", "331"]


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_enum(model, mode):
    class Options(enum.Enum):
        A = "A"
        B = "B"
        C = "C"

    client = instructor.from_vertexai(gm.GenerativeModel(model), mode)

    response = client.create(
        response_model=Options,
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert response in [Options.A, Options.B, Options.C]


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_bool(model, mode):
    client = instructor.from_vertexai(gm.GenerativeModel(model), mode)

    response = client.create(
        response_model=bool,
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert type(response) == bool
