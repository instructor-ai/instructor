# Single-Label Classification

This example demonstrates how to perform single-label classification using the OpenAI API. The example uses the `gpt-3.5-turbo` model to classify text as either `SPAM` or `NOT_SPAM`.

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
