from typing import Literal

import pytest
import instructor

from pydantic import BaseModel


class SinglePrediction(BaseModel):
    """
    Correct class label for the given text
    """

    class_label: Literal["spam", "not_spam"]


data = [
    ("I am a spammer", "spam"),
    ("I am not a spammer", "not_spam"),
]


@pytest.mark.parametrize("data", data)
def test_classification(data):
    client = instructor.from_provider("google/gemini-2.0-flash-exp")

    input, expected = data
    resp = client.chat.completions.create(
        response_model=SinglePrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {input}",
            },
        ],
    )
    assert resp.class_label == expected


# Adjust the prediction model to accommodate a list of labels
class MultiClassPrediction(BaseModel):
    predicted_labels: list[Literal["billing", "general_query", "hardware"]]


data = [
    (
        "I am having trouble with my billing",
        ["billing"],
    ),
    (
        "I am having trouble with my hardware",
        ["hardware"],
    ),
    (
        "I have a general query and a billing issue",
        ["general_query", "billing"],
    ),
]


@pytest.mark.parametrize("data", data)
def test_multi_classify(data):
    client = instructor.from_provider("google/gemini-2.0-flash-exp")

    input, expected = data

    resp = client.chat.completions.create(
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following support ticket: {input}",
            },
        ],
    )
    assert set(resp.predicted_labels) == set(expected)
