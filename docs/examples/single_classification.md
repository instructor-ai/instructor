---
title: Single-Label Classification with OpenAI API
description: Learn to implement single-label classification using the OpenAI API to classify text as SPAM or NOT_SPAM.
---

# Single-Label Classification

This example demonstrates how to perform single-label classification using the OpenAI API. The example uses the `gpt-3.5-turbo` model to classify text as either `SPAM` or `NOT_SPAM`.

```python
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI
import instructor

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_openai(OpenAI())


class ClassificationResponse(BaseModel):
    """
    A few-shot example of text classification:

    Examples:
    - "Buy cheap watches now!": SPAM
    - "Meeting at 3 PM in the conference room": NOT_SPAM
    - "You've won a free iPhone! Click here": SPAM
    - "Can you pick up some milk on your way home?": NOT_SPAM
    - "Increase your followers by 10000 overnight!": SPAM
    """

    label: Literal["SPAM", "NOT_SPAM"] = Field(
        ...,
        description="The predicted class label.",
    )


def classify(data: str) -> ClassificationResponse:
    """Perform single-label classification on the input text."""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=ClassificationResponse,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: <text>{data}</text>",
            },
        ],
    )


if __name__ == "__main__":
    for text, label in [
        ("Hey Jason! You're awesome", "NOT_SPAM"),
        ("I am a nigerian prince and I need your help.", "SPAM"),
    ]:
        prediction = classify(text)
        assert prediction.label == label
        print(f"Text: {text}, Predicted Label: {prediction.label}")
        #> Text: Hey Jason! You're awesome, Predicted Label: NOT_SPAM
        #> Text: I am a nigerian prince and I need your help., Predicted Label: SPAM
```
