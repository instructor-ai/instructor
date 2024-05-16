from itertools import product
from typing import Literal

import pytest
import instructor

from pydantic import BaseModel

from instructor.function_calls import Mode
from ..util import models, modes


class SinglePrediction(BaseModel):
    """
    Correct class label for the given text
    """

    class_label: Literal["spam", "not_spam"]


data = [
    ("I am a spammer", "spam"),
    ("I am not a spammer", "not_spam"),
]


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
@pytest.mark.asyncio
async def test_classification(model, data, mode, aclient):
    client = instructor.from_openai(aclient, mode=mode)

    if mode == instructor.Mode.JSON and model in {"gpt-3.5-turbo", "gpt-4"}:
        pytest.skip(
            "JSON mode is not supported for gpt-3.5-turbo and gpt-4, skipping test"
        )

    input, expected = data
    resp = await client.chat.completions.create(
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


@pytest.mark.parametrize("model, data, mode", product(models, data, modes))
@pytest.mark.asyncio
async def test_multi_classify(model, data, mode, aclient):
    client = instructor.from_openai(aclient, mode=mode)

    if (mode, model) in {
        (Mode.JSON, "gpt-3.5-turbo"),
        (Mode.JSON, "gpt-4"),
    }:
        pytest.skip(f"{mode} mode is not supported for {model}, skipping test")

    input, expected = data

    resp = await client.chat.completions.create(
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
