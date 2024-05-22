import instructor
import enum

import google.generativeai as genai
from typing import Literal, Union


def test_literal():
    client = instructor.from_gemini(
        genai.GenerativeModel("models/gemini-1.5-flash-latest")
    )

    response = client.chat.completions.create(
        response_model=Literal["1231", "212", "331"],
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert response in ["1231", "212", "331"]


def test_union():
    client = instructor.from_gemini(
        genai.GenerativeModel("models/gemini-1.5-flash-latest")
    )

    response = client.chat.completions.create(
        response_model=Union[int, str],
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert type(response) in [int, str]


def test_enum():
    class Options(enum.Enum):
        A = "A"
        B = "B"
        C = "C"

    client = instructor.from_gemini(
        genai.GenerativeModel("models/gemini-1.5-flash-latest")
    )

    response = client.chat.completions.create(
        response_model=Options,
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert response in [Options.A, Options.B, Options.C]


def test_bool():
    client = instructor.from_gemini(
        genai.GenerativeModel("models/gemini-1.5-flash-latest")
    )

    response = client.chat.completions.create(
        response_model=bool,
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert type(response) == bool
