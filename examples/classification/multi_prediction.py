from typing import List
import enum
import openai
from pydantic import BaseModel
from instructor import patch

patch()


# Define new Enum class for multiple labels
class MultiLabels(str, enum.Enum):
    TECH_ISSUE = "tech_issue"
    BILLING = "billing"
    GENERAL_QUERY = "general_query"


# Adjust the prediction model to accommodate a list of labels
class MultiClassPrediction(BaseModel):
    """
    List of correct class labels for the given text (Multi Class)
    """

    class_labels: List[MultiLabels]


# Modify the classify function
def multi_classify(data: str) -> MultiClassPrediction:
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following support ticket: {data}",
            },
        ],
    )  # type: ignore


# Example using a support ticket
ticket = "My account is locked and I can't access my billing info."
prediction = multi_classify(ticket)
assert MultiLabels.TECH_ISSUE in prediction.class_labels
assert MultiLabels.BILLING in prediction.class_labels
