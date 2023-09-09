# Example: Text Classification using OpenAI and Pydantic

This tutorial showcases how to implement text classification tasks—specifically, single-label and multi-label classifications—using the OpenAI API, Python's **`enum`** module, and Pydantic models.

!!! tips "Motivation"

    Text classification is a common problem in many NLP applications, such as spam detection or support ticket categorization. The goal is to provide a systematic way to handle these cases using OpenAI's GPT models in combination with Python data structures.

## Single-Label Classification

### Defining the Structures

For single-label classification, we first define an **`enum`** for possible labels and a Pydantic model for the output.

```python
import enum
from pydantic import BaseModel

class Labels(str, enum.Enum):
    """Enumeration for single-label text classification."""
    SPAM = "spam"
    NOT_SPAM = "not_spam"

class SinglePrediction(BaseModel):
    """
    Class for a single class label prediction.
    """
    class_label: Labels

```

### Classifying Text

The function **`classify`** will perform the single-label classification.

```python
import openai
import instructor

# Patch the OpenAI API to use the `ChatCompletion`
# endpoint with `response_model` enabled.
instructor.patch() 

def classify(data: str) -> SinglePrediction:
    """Perform single-label classification on the input text."""
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
```

### Testing and Evaluation

Let's run an example to see if it correctly identifies a spam message.

```python
# Test single-label classification
prediction = classify("Hello there I'm a Nigerian prince and I want to give you money")
assert prediction.class_label == Labels.SPAM

```

## Multi-Label Classification

### Defining the Structures

For multi-label classification, we introduce a new enum class and a different Pydantic model to handle multiple labels.

```python
# Define Enum class for multiple labels
class MultiLabels(str, enum.Enum):
    TECH_ISSUE = "tech_issue"
    BILLING = "billing"
    GENERAL_QUERY = "general_query"

# Define the multi-class prediction model
class MultiClassPrediction(BaseModel):
    """
    Class for a multi-class label prediction.
    """
    class_labels: List[MultiLabels]

```

### Classifying Text

The function **`multi_classify`** is responsible for multi-label classification.

```python
def multi_classify(data: str) -> MultiClassPrediction:
    """Perform multi-label classification on the input text."""
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

```

### Testing and Evaluation

Finally, we test the multi-label classification function using a sample support ticket.

```python
# Test multi-label classification
ticket = "My account is locked and I can't access my billing info."
prediction = multi_classify(ticket)
assert MultiLabels.TECH_ISSUE in prediction.class_labels
assert MultiLabels.BILLING in prediction.class_labels
```