# Instructor Classify 

A **fluent, type‑safe API** for text classification built on top of
[Instructor](https://github.com/jxnl/instructor/) and the OpenAI Chat
Completion API.

> Both synchronous and asynchronous APIs are fully supported.

---

## Why text classification?

* **Intent classification** – route user queries to the right piece of business
  logic (e.g. support, sales, billing).
* **Content moderation** – detect harmful or disallowed text before it reaches
  users.
* **Data segmentation** – attach structured tags to free‑form text for
  analytics or search.

---

## Installation

```bash
pip install instructor openai pydantic
```

Set the `OPENAI_API_KEY` environment variable to your key.

---

## 1 · Define your labels

### a) YAML (recommended)

Non‑developers can tweak labels without touching code.

Including *positive* and *negative* examples for every label lets the model see
real data and generally yields **higher accuracy**:

```yaml
# labels.yaml
system_message: |
  You are an expert classification system designed to analyse user inputs and
  categorise them into predefined labels.
label_definitions:
  - label: question
    description: The user asks for information or clarification.
    examples:
      examples_positive:
        - "What is the capital of France?"
        - "How does photosynthesis work?"
      examples_negative:
        - "Please book me a flight to Paris."
  - label: request
    description: The user wants the assistant to perform an action.
  - label: scheduling
    description: The user is arranging a meeting or event.
  - label: coding
    description: The user needs help writing or understanding code.
```

Load it with one line of Python:

```python
from schema import ClassificationDefinition

definition = ClassificationDefinition.from_yaml("labels.yaml")
```

### b) Pure Python (inline fall‑back)

```python
from schema import ClassificationDefinition, LabelDefinition

definition = ClassificationDefinition(
    system_message="You are an expert classification system …",
    label_definitions=[
        LabelDefinition(label="question", description="User asks for info."),
        LabelDefinition(label="request", description="User wants an action."),
        # …
    ],
)
```

---

## 2 · Create the classifier

```python
from classify import Classifier
from openai import OpenAI

classifier = (
    Classifier(definition)
    .with_client(OpenAI())     # Wraps client with `instructor.from_openai`
    .with_model("gpt-4o-mini") # Swap to "gpt-4o" or "gpt-3.5-turbo" at will
)
```

---

## 3 · Predict

### Single‑label

```python
result = classifier.predict("What is machine learning?")
print(result.label)  # -> "question"
```

### Multi‑label

```python
multi = classifier.predict_multi("Schedule a meeting and help me with Python.")
print(multi.labels)  # -> ["scheduling", "coding"]
```

The return objects are **dynamically generated Pydantic models**, so editors
like VS Code provide auto‑completion and type checking.

---

## 4 · Under the hood

* **Prompt templating** – Jinja templates include examples and label metadata.
* **Schema‑driven models** – `ClassificationDefinition` builds Enum‑safe
  Pydantic models for single and multi classification.
* **Instructor** – intercepts the API call and validates the response against
  the generated model.

---

## Async Support

```python
from classify import AsyncClassifier
from openai import AsyncOpenAI

# Create an async classifier with AsyncOpenAI client
async_classifier = (
    AsyncClassifier(definition)
    .with_client(AsyncOpenAI())
    .with_model("gpt-4o-mini")
)

# Async prediction
result = await async_classifier.predict("What is machine learning?")
print(result.label)  # -> "question"

# Async batch prediction
results = await async_classifier.batch_predict(["What is ML?", "Help me with Python"], n_jobs=2)
```

## Roadmap

- [x] Async support
- [ ] Streaming classification

Contributions are welcome! Feel free to open an issue or pull request.

