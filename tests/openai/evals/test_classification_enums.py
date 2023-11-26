import enum
from itertools import product
from typing import List

import pytest
import instructor
from openai import OpenAI

from pydantic import BaseModel


class Labels(str, enum.Enum):
    SPAM = "spam"
    NOT_SPAM = "not_spam"


class SinglePrediction(BaseModel):
    """
    Correct class label for the given text
    """

    class_label: Labels


models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]
modes = [instructor.Mode.FUNCTIONS, instructor.Mode.JSON, instructor.Mode.TOOLS]
data = [
    (
        "I am a spammer",
        Labels.SPAM,
    ),
    (
        "I am not a spammer",
        Labels.NOT_SPAM,
    ),
]


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
def test_classification(model, data, mode):
    client = instructor.patch(OpenAI(), mode=mode)

    if mode == instructor.Mode.JSON and model in {"gpt-3.5-turbo", "gpt-4"}:
        pytest.skip(
            "JSON mode is not supported for gpt-3.5-turbo and gpt-4, skipping test"
        )

    input, expected = data
    resp = client.chat.completions.create(
        model=model,
        response_model=SinglePrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {input}",
            },
        ],
    )
    assert resp.class_label == expected


# Define new Enum class for multiple labels
class MultiLabels(str, enum.Enum):
    BILLING = "billing"
    GENERAL_QUERY = "general_query"
    HARDWARE = "hardware"


# Adjust the prediction model to accommodate a list of labels
class MultiClassPrediction(BaseModel):
    predicted_labels: List[MultiLabels]


data = [
    (
        "I am having trouble with my billing",
        [MultiLabels.BILLING],
    ),
    (
        "I am having trouble with my hardware",
        [MultiLabels.HARDWARE],
    ),
    (
        "I have a general query and a billing issue",
        [MultiLabels.GENERAL_QUERY, MultiLabels.BILLING],
    ),
]


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
def test_multi_classify(model, data, mode):
    client = instructor.patch(OpenAI(), mode=mode)

    if mode == instructor.Mode.JSON and model in {"gpt-3.5-turbo", "gpt-4"}:
        pytest.skip(
            "JSON mode is not supported for gpt-3.5-turbo and gpt-4, skipping test"
        )

    input, expected = data

    resp = client.chat.completions.create(
        model=model,
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following support ticket: {input}",
            },
        ],
    )
    assert set(resp.predicted_labels) == set(expected)
