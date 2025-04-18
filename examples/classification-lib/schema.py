from __future__ import annotations
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal
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
