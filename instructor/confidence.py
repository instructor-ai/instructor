"""
Confidence Scoring for LLM Extractions

Provides TRUE confidence scores by analyzing token log probabilities
from LLM responses. Zero extra API calls - just parses existing data.

Author: Ruthvik Bandari
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, TypeVar
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Confidence level interpretation."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class FieldConfidence:
    """Confidence score for a single extracted field."""

    field_name: str
    value: Any
    confidence: float
    level: ConfidenceLevel
    token_count: int = 0
    avg_logprob: float = 0.0

    def __post_init__(self):
        if self.confidence >= 0.90:
            self.level = ConfidenceLevel.HIGH
        elif self.confidence >= 0.75:
            self.level = ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.50:
            self.level = ConfidenceLevel.LOW
        else:
            self.level = ConfidenceLevel.VERY_LOW


@dataclass
class ConfidenceResult:
    """Complete confidence analysis for an extraction."""

    overall: float
    level: ConfidenceLevel
    fields: dict[str, FieldConfidence]
    token_count: int
    processing_time_ms: float
    model: str = ""

    @property
    def is_reliable(self) -> bool:
        return self.level == ConfidenceLevel.HIGH

    @property
    def low_confidence_fields(self) -> list[str]:
        return [
            name
            for name, fc in self.fields.items()
            if fc.level in (ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "level": self.level.value,
            "is_reliable": self.is_reliable,
            "low_confidence_fields": self.low_confidence_fields,
            "token_count": self.token_count,
            "processing_time_ms": self.processing_time_ms,
            "model": self.model,
            "fields": {
                name: {
                    "value": fc.value,
                    "confidence": fc.confidence,
                    "level": fc.level.value,
                }
                for name, fc in self.fields.items()
            },
        }


class ConfidenceScorer:
    """
    Calculates confidence scores from LLM token logprobs.

    Zero API calls - uses existing response data.
    < 1ms processing time.
    Zero additional dependencies.
    """

    def __init__(
        self,
        high_threshold: float = 0.90,
        medium_threshold: float = 0.75,
        low_threshold: float = 0.50,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold

    def logprob_to_probability(self, logprob: float) -> float:
        try:
            return math.exp(logprob)
        except (OverflowError, ValueError):
            return 0.0

    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        if confidence >= self.high_threshold:
            return ConfidenceLevel.HIGH
        elif confidence >= self.medium_threshold:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.low_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def extract_logprobs_openai(self, response: Any) -> list[dict]:
        tokens = []
        try:
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "logprobs") and choice.logprobs:
                    logprobs_data = choice.logprobs
                    if hasattr(logprobs_data, "content") and logprobs_data.content:
                        for item in logprobs_data.content:
                            if hasattr(item, "token") and hasattr(item, "logprob"):
                                if item.logprob is not None:
                                    prob = self.logprob_to_probability(item.logprob)
                                    tokens.append(
                                        {
                                            "token": item.token,
                                            "logprob": item.logprob,
                                            "probability": prob,
                                        }
                                    )
        except Exception:
            pass
        return tokens

    def map_tokens_to_fields(
        self, tokens: list[dict], extracted_data: dict[str, Any]
    ) -> dict[str, list[dict]]:
        field_tokens: dict[str, list[dict]] = {k: [] for k in extracted_data.keys()}

        if not tokens:
            return field_tokens

        full_text = "".join(t["token"] for t in tokens)

        for field_name, value in extracted_data.items():
            if value is None:
                continue

            value_str = str(value).strip()
            if not value_str:
                continue

            value_lower = value_str.lower()
            text_lower = full_text.lower()

            start_idx = text_lower.find(value_lower)
            if start_idx == -1 and isinstance(value, (int, float)):
                start_idx = text_lower.find(str(value))

            if start_idx != -1:
                char_count = 0
                for token_data in tokens:
                    token_start = char_count
                    token_end = char_count + len(token_data["token"])
                    value_end = start_idx + len(value_str)
                    if token_start < value_end and token_end > start_idx:
                        field_tokens[field_name].append(token_data)
                    char_count = token_end

            if not field_tokens[field_name]:
                avg_prob = (
                    sum(t["probability"] for t in tokens) / len(tokens)
                    if tokens
                    else 0.5
                )
                field_tokens[field_name] = [
                    {"probability": avg_prob, "token": "", "logprob": 0}
                ]

        return field_tokens

    def calculate_field_confidence(
        self, field_name: str, value: Any, tokens: list[dict]
    ) -> FieldConfidence:
        if not tokens:
            return FieldConfidence(
                field_name=field_name,
                value=value,
                confidence=0.5,
                level=ConfidenceLevel.LOW,
                token_count=0,
                avg_logprob=0.0,
            )

        probabilities = [t["probability"] for t in tokens]
        logprobs = [t.get("logprob", 0) for t in tokens]

        if probabilities:
            log_sum = sum(math.log(max(p, 1e-10)) for p in probabilities)
            confidence = math.exp(log_sum / len(probabilities))
        else:
            confidence = 0.5

        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0

        return FieldConfidence(
            field_name=field_name,
            value=value,
            confidence=round(confidence, 4),
            level=self.get_confidence_level(confidence),
            token_count=len(tokens),
            avg_logprob=round(avg_logprob, 4),
        )

    def score(
        self, response: Any, extracted_data: dict[str, Any], model: str = ""
    ) -> ConfidenceResult:
        start_time = time.perf_counter()

        tokens = self.extract_logprobs_openai(response)
        field_tokens = self.map_tokens_to_fields(tokens, extracted_data)

        field_results: dict[str, FieldConfidence] = {}
        for field_name, value in extracted_data.items():
            field_results[field_name] = self.calculate_field_confidence(
                field_name, value, field_tokens.get(field_name, [])
            )

        if field_results:
            confidences = [fc.confidence for fc in field_results.values()]
            overall = sum(confidences) / len(confidences)
        else:
            overall = 0.0

        processing_time = (time.perf_counter() - start_time) * 1000

        return ConfidenceResult(
            overall=round(overall, 4),
            level=self.get_confidence_level(overall),
            fields=field_results,
            token_count=len(tokens),
            processing_time_ms=round(processing_time, 3),
            model=model,
        )


_default_scorer = ConfidenceScorer()


def score_confidence(
    response: Any, extracted_data: dict[str, Any], model: str = ""
) -> ConfidenceResult:
    """Calculate confidence scores from an LLM response."""
    return _default_scorer.score(response, extracted_data, model)


def enable_logprobs(kwargs: dict) -> dict:
    """Enable logprobs in create() kwargs."""
    kwargs["logprobs"] = True
    return kwargs


T = TypeVar("T")


class ConfidenceError(Exception):
    """Raised when confidence scoring fails."""

    pass


class LowConfidenceError(Exception):
    """Raised when extraction confidence is below threshold."""

    def __init__(
        self, confidence: ConfidenceResult, threshold: float, message: str | None = None
    ):
        self.confidence = confidence
        self.threshold = threshold
        self.message = message or (
            f"Extraction confidence {confidence.overall:.2%} is below "
            f"threshold {threshold:.2%}. "
            f"Low confidence fields: {confidence.low_confidence_fields}"
        )
        super().__init__(self.message)
