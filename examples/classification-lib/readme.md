# Instructor Classify 

A **fluent, type-safe API** for text classification built on top of
[Instructor](https://github.com/jxnl/instructor/) with support for multiple LLM providers.

> Both synchronous and asynchronous APIs are fully supported with comprehensive evaluation tools.

---

## Why text classification?

* **Intent classification** – route user queries to the right piece of business
  logic (e.g. support, sales, billing).
* **Content moderation** – detect harmful or disallowed text before it reaches
  users.
* **Data segmentation** – attach structured tags to free-form text for
  analytics or search.

---

## Installation

```bash
pip install instructor pydantic
```

Install your preferred LLM provider package:
```bash
pip install openai  # For OpenAI
# Or: pip install anthropic, google-generativeai, etc.
```

Set the appropriate API key environment variable (e.g., `OPENAI_API_KEY`).

---

## 1 · Define your labels

### a) YAML (recommended)

Non-developers can tweak labels without touching code.

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

### b) Pure Python (inline fall-back)

```python
from schema import ClassificationDefinition, LabelDefinition

definition = ClassificationDefinition(
    system_message="You are an expert classification system ...",
    label_definitions=[
        LabelDefinition(label="question", description="User asks for info."),
        LabelDefinition(label="request", description="User wants an action."),
        # ...
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
    .with_model("gpt-4o-mini") # Swap models easily
)
```

Works with multiple LLM providers through Instructor:
```python
# Anthropic
from anthropic import Anthropic
classifier = (
    Classifier(definition)
    .with_client(Anthropic())
    .with_model("claude-3-haiku-20240307")
)

# Google Gemini
from google.generativeai import GenerativeModel
classifier = (
    Classifier(definition)
    .with_client(GenerativeModel("gemini-pro"))
)
```

---

## 3 · Predict

### Single-label

```python
result = classifier.predict("What is machine learning?")
print(result.label)  # -> "question"
```

### Multi-label

```python
multi = classifier.predict_multi("Schedule a meeting and help me with Python.")
print(multi.labels)  # -> ["scheduling", "coding"]
```

### Batch processing

```python
# Batch prediction with multiple threads
results = classifier.batch_predict(
    ["What is ML?", "Help with Python", "Book a meeting"],
    n_jobs=3,  # Process 3 items concurrently
    show_progress=True  # Display a progress bar
)
```

The return objects are **dynamically generated Pydantic models**, so editors
like VS Code provide auto-completion and type checking.

---

## 4 · Evaluation

Comprehensive tools for evaluating classifier performance:

```python
from schema import EvalSet
from classify import Classifier
from eval import evaluate_classifier, display_evaluation_results

# Load an evaluation set from YAML
eval_set = EvalSet.from_yaml("tests/example_evalset.yaml")

# Run evaluation
result = evaluate_classifier(classifier, eval_set, n_jobs=4)

# Display results with Rich
display_evaluation_results(result, detailed=True)
```

Example YAML format for evaluation sets:

```yaml
name: "Example Classification Evaluation Set"
description: "Test set for evaluating the classifier"
classification_type: "single"  # or "multi"
examples:
  - text: "What is the capital of France?"
    expected_label: "question"
  
  - text: "Schedule a meeting for tomorrow at 3pm."
    expected_label: "scheduling"
```

For multi-label evaluation:
```yaml
classification_type: "multi"
examples:
  - text: "Help me schedule a coding session."
    expected_labels: ["scheduling", "coding"]
```

---

## 5 · Advanced Evaluation Harness

A comprehensive framework for comparing models across datasets is included:

```bash
cd eval_harness
python unified_eval.py --config configs/unified_config.yaml
```

Features:
- **Multi-model comparison** across datasets
- **Cost and latency tracking** for budget analysis
- **Statistical reliability** with bootstrapped confidence intervals
- **Error pattern analysis** and confusion matrices
- **Rich visualizations** and comprehensive reports

Configuration via YAML:
```yaml
# Models to evaluate
models:
  - "gpt-3.5-turbo"
  - "gpt-4o-mini"

# Evaluation datasets
eval_sets:
  - "datasets/custom_evalset.yaml"
  - "datasets/complex_evalset.yaml"

# Analysis parameters
bootstrap_samples: 1000
confidence_level: 0.95
n_jobs: 4
```

---

## 6 · Async Support

```python
from classify import AsyncClassifier
from openai import AsyncOpenAI

# Create an async classifier
async_classifier = (
    AsyncClassifier(definition)
    .with_client(AsyncOpenAI())
    .with_model("gpt-4o-mini")
)

# Async prediction
result = await async_classifier.predict("What is machine learning?")
print(result.label)  # -> "question"

# Async batch prediction with concurrency control
results = await async_classifier.batch_predict(
    ["What is ML?", "Help me with Python"],
    n_jobs=2,
    max_concurrency=5  # Limit concurrent API calls
)
```

---

## 7 · Under the hood

* **Prompt templating** – Jinja templates include examples and label metadata
* **Schema-driven models** – `ClassificationDefinition` builds Enum-safe
  Pydantic models for single and multi classification
* **Provider-agnostic** – Works with any LLM provider supported by Instructor
* **Concurrency controls** – Manage API rate limits with granular settings
* **Resource optimization** – Efficient batch processing and progress tracking

---

## Roadmap

- [x] Async support
- [x] Evaluation framework with metrics
- [x] Advanced evaluation harness
- [x] Multi-label classification
- [x] Multi-provider support
- [ ] Streaming classification

Contributions are welcome! Feel free to open an issue or pull request.