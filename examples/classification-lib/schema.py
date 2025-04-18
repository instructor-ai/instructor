from __future__ import annotations
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal, TypeVar, Generic
import yaml


class Examples(BaseModel):
    examples_positive: list[str] | None = None
    examples_negative: list[str] | None = None


class LabelDefinition(BaseModel):
    label: str
    description: str
    examples: Examples | None = None

    @field_validator("label")
    @classmethod
    def lowercase_label(cls, value: str) -> str:
        return value.lower()


class ClassificationDefinition(BaseModel):
    system_message: str | None = None
    label_definitions: list[LabelDefinition]

    @model_validator(mode="after")
    def validate_unique_labels(cls, model):
        labels = [ld.label for ld in model.label_definitions]
        if len(labels) != len(set(labels)):
            raise ValueError("Label definitions must have unique labels")
        return model

    @classmethod
    def from_yaml(cls, yaml_path: str) -> ClassificationDefinition:
        with open(yaml_path) as file:
            yaml_data = yaml.safe_load(file)
        return cls(**yaml_data)

    # ------------------------------------------------------------------
    # Dynamic Enum & Pydantic model helpers
    # ------------------------------------------------------------------

    def get_classification_model(self) -> type[BaseModel]:
        labels = tuple(ld.label for ld in self.label_definitions)

        class ClassificationModel(BaseModel):
            """Single‑label classification model. Only use a single label from the provided examples."""

            label: Literal[labels]  # type: ignore

        return ClassificationModel

    def get_multiclassification_model(self) -> type[BaseModel]:
        labels = tuple(ld.label for ld in self.label_definitions)

        class MultiClassificationModel(BaseModel):
            """Multi‑label classification model. Only use labels from the provided examples."""

            labels: list[Literal[labels]]  # type: ignore

        return MultiClassificationModel


# Type variable to represent the prediction model (single or multi-label)
T = TypeVar("T", bound=BaseModel)


class EvalExample(BaseModel):
    """An example for evaluation with text and expected label(s)."""

    text: str
    expected_label: str | None = None  # For single-label classification
    expected_labels: list[str] | None = None  # For multi-label classification


class EvalSet(BaseModel, Generic[T]):
    """A set of examples for evaluating classifier performance."""

    name: str
    description: str
    examples: list[EvalExample]
    classification_type: Literal["single", "multi"]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> EvalSet:
        """Load evaluation set from YAML file."""
        with open(yaml_path) as file:
            yaml_data = yaml.safe_load(file)
        return cls(**yaml_data)

    def validate_against_definition(self, definition: ClassificationDefinition) -> bool:
        """Validate that all expected labels are in the classification definition."""
        valid_labels = {ld.label for ld in definition.label_definitions}

        for example in self.examples:
            if example.expected_label and example.expected_label not in valid_labels:
                return False
            if example.expected_labels and any(
                label not in valid_labels for label in example.expected_labels
            ):
                return False

        return True


class EvalResult(BaseModel):
    """Results from evaluating a classifier on an evaluation set."""

    eval_set_name: str
    total_examples: int
    correct_predictions: int
    accuracy: float
    per_label_metrics: dict[str, dict[str, float]]  # label -> {precision, recall, f1}
    confusion_matrix: dict[
        str, dict[str, int]
    ]  # actual label -> {predicted label -> count}

    # Store prediction details for further analysis
    predictions: list[dict[str, str | list[str]]]  # list of {text, expected, predicted}
