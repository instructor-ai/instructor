import enum
import openai
from pydantic import BaseModel
from instructor import patch

patch()


class Labels(str, enum.Enum):
    SPAM = "spam"
    NOT_SPAM = "not_spam"


class SinglePrediction(BaseModel):
    """
    Correct class label for the given text
    """

    class_label: Labels


def classify(data: str) -> SinglePrediction:
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        response_model=SinglePrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {data}",
            },
        ],
    )  # type: ignore


prediction = classify("Hello there I'm a nigerian prince and I want to give you money")
assert prediction.class_label == Labels.SPAM
