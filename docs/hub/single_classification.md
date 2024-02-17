# Single-Label Classification

This tutorial showcases how to implement text classification tasks—specifically, single-label and multi-label classifications—using the OpenAI API.

```python
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI
import instructor

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.patch(OpenAI())


class ClassificationResponse(BaseModel):
    label: Literal["SPAM", "NOT_SPAM"] = Field(
        ...,
        description="The predicted class label.",
    )


def classify(data: str) -> ClassificationResponse:
    """Perform single-label classification on the input text."""
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=ClassificationResponse,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {data}",
            },
        ],
    )


if __name__ == "__main__":
    data = "Hey Jason! You're awesome"
    prediction = classify(data)
    assert prediction.label == "NOT_SPAM"
    print(prediction)
    #> label='NOT_SPAM'

    data = "I am a nigerian prince and I need your help."
    prediction = classify(data)
    assert prediction.label == "SPAM"
    print(prediction)
    #> label='SPAM'
```
