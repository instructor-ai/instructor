from __future__ import annotations

from openai import OpenAI

from schema import ClassificationDefinition, LabelDefinition  # type: ignore
from classify import Classifier  # type: ignore

# ---------------------------------------------------------------------------
# Load the YAML schema and build helper types
# ---------------------------------------------------------------------------


# Load the YAML schema
definition = ClassificationDefinition(
    system_message="You are an expert classification system designed to analyze user inputs and categorize them into predefined labels. Your task is to carefully examine each input and determine which category it belongs to based on the provided definitions and examples. Focus on the semantic meaning and intent behind the text, not just keywords. Provide accurate classifications even for ambiguous or edge cases.",
    label_definitions=[
        LabelDefinition(
            label="question",
            description="The user is asking for information or clarification on a topic.",
        ),
        LabelDefinition(
            label="request",
            description="The user is asking for the assistant to perform a task or take action.",
        ),
        LabelDefinition(
            label="scheduling",
            description="The user is asking for the assistant to schedule a task or event.",
        ),
        LabelDefinition(
            label="coding",
            description="The user is asking for the assistant to write code or help with a coding problem.",
        ),
    ],
)
# Instantiate classifier (type checkers may not infer dynamic model precisely)
classifier = Classifier(definition).with_client(OpenAI())  # type: ignore

# ---------------------------------------------------------------------------
# Single‑label prediction
# ---------------------------------------------------------------------------

result = classifier.predict("What is machine learning?")
# -> VSCode / Pyright should infer ``result`` as ``PredictionModel``
print("Predicted label:", result.label)
assert result.label == "question"

# ---------------------------------------------------------------------------
# Multi‑label prediction
# ---------------------------------------------------------------------------

multi = classifier.predict_multi("Schedule a meeting and help me with Python code.")
assert multi.labels == ["scheduling", "coding"]
