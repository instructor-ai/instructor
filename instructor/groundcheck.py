"""
GroundCheck: Source Grounding Verification for LLM Extractions

This module provides tools to verify whether data extracted by LLMs
is actually grounded in the source text, helping detect hallucinations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class VerificationMethod(str, Enum):
    """Methods used to verify field grounding."""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    NUMERIC_MATCH = "numeric_match"
    SEMANTIC_MATCH = "semantic_match"
    NOT_FOUND = "not_found"


@dataclass
class FieldResult:
    """Result of verifying a single field against source text."""
    field_name: str
    value: Any
    confidence: float
    method: VerificationMethod
    evidence: Optional[str] = None
    source_span: Optional[Tuple[int, int]] = None
    flagged: bool = False
    reason: Optional[str] = None
    
    def __post_init__(self):
        if self.confidence < 0.5 and not self.flagged:
            self.flagged = True


@dataclass
class VerificationResult:
    """Complete result of verifying extracted data against source."""
    source_text: str
    extracted_data: Dict[str, Any]
    field_results: Dict[str, FieldResult] = field(default_factory=dict)
    overall_confidence: float = 0.0
    flagged_fields: List[str] = field(default_factory=list)
    is_reliable: bool = True
    verification_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.field_results:
            confidences = [r.confidence for r in self.field_results.values()]
            self.overall_confidence = sum(confidences) / len(confidences)
            self.flagged_fields = [
                name for name, result in self.field_results.items() 
                if result.flagged
            ]
            self.is_reliable = len(self.flagged_fields) == 0


class GroundCheck:
    """
    Verifies whether LLM-extracted data is grounded in source text.
    
    Uses multiple strategies: exact match, fuzzy match, numeric match,
    and optional semantic matching.
    
    Example:
        gc = GroundCheck()
        result = gc.verify(
            source_text="Invoice #12345 from Acme Corp",
            extracted_data={"invoice": "12345", "vendor": "Acme Corp"}
        )
        print(result.flagged_fields)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        fuzzy_threshold: float = 0.85,
        enable_semantic: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.confidence_threshold = confidence_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.enable_semantic = enable_semantic and EMBEDDINGS_AVAILABLE
        self._embedding_model = None
        if self.enable_semantic:
            self._embedding_model = SentenceTransformer(embedding_model)
    
    def verify(
        self,
        source_text: str,
        extracted_data: Dict[str, Any],
        field_thresholds: Optional[Dict[str, float]] = None,
    ) -> VerificationResult:
        """Verify extracted data against source text."""
        import time
        start_time = time.time()
        field_thresholds = field_thresholds or {}
        field_results: Dict[str, FieldResult] = {}
        for field_name, value in extracted_data.items():
            threshold = field_thresholds.get(field_name, self.confidence_threshold)
            result = self._verify_field(source_text, field_name, value, threshold)
            field_results[field_name] = result
        verification_time_ms = (time.time() - start_time) * 1000
        return VerificationResult(
            source_text=source_text,
            extracted_data=extracted_data,
            field_results=field_results,
            verification_time_ms=verification_time_ms,
        )
    
    def _verify_field(self, source_text: str, field_name: str, value: Any, threshold: float) -> FieldResult:
        """Verify a single field value against source text."""
        if value is None:
            return FieldResult(field_name=field_name, value=value, confidence=0.0, method=VerificationMethod.NOT_FOUND, flagged=True, reason="Value is None")
        if isinstance(value, (list, dict)):
            return self._verify_complex_field(source_text, field_name, value, threshold)
        str_value = str(value).strip()
        if not str_value:
            return FieldResult(field_name=field_name, value=value, confidence=0.0, method=VerificationMethod.NOT_FOUND, flagged=True, reason="Empty value")
        exact_result = self._exact_match(source_text, str_value)
        if exact_result[0] >= 0.95:
            return FieldResult(field_name=field_name, value=value, confidence=exact_result[0], method=VerificationMethod.EXACT_MATCH, evidence=exact_result[1], source_span=exact_result[2], flagged=exact_result[0] < threshold)
        if isinstance(value, (int, float)) or self._looks_like_number(str_value):
            numeric_result = self._numeric_match(source_text, value)
            if numeric_result[0] >= 0.8:
                return FieldResult(field_name=field_name, value=value, confidence=numeric_result[0], method=VerificationMethod.NUMERIC_MATCH, evidence=numeric_result[1], flagged=numeric_result[0] < threshold)
        if RAPIDFUZZ_AVAILABLE:
            fuzzy_result = self._fuzzy_match(source_text, str_value)
            if fuzzy_result[0] >= self.fuzzy_threshold:
                return FieldResult(field_name=field_name, value=value, confidence=fuzzy_result[0], method=VerificationMethod.FUZZY_MATCH, evidence=fuzzy_result[1], flagged=fuzzy_result[0] < threshold)
        if self.enable_semantic and self._embedding_model is not None:
            semantic_result = self._semantic_match(source_text, str_value)
            if semantic_result[0] >= 0.7:
                return FieldResult(field_name=field_name, value=value, confidence=semantic_result[0], method=VerificationMethod.SEMANTIC_MATCH, evidence=semantic_result[1], flagged=semantic_result[0] < threshold, reason="Semantic similarity match")
        best_fuzzy = self._fuzzy_match(source_text, str_value) if RAPIDFUZZ_AVAILABLE else (0.0, None)
        return FieldResult(field_name=field_name, value=value, confidence=best_fuzzy[0] if best_fuzzy[0] > 0.3 else 0.0, method=VerificationMethod.NOT_FOUND, evidence=best_fuzzy[1] if best_fuzzy[0] > 0.3 else None, flagged=True, reason=f"Value '{str_value}' not found in source text")
    
    def _exact_match(self, source: str, value: str) -> Tuple[float, Optional[str], Optional[Tuple[int, int]]]:
        """Check for exact match (case-insensitive)."""
        source_lower = source.lower()
        value_lower = value.lower()
        idx = source_lower.find(value_lower)
        if idx != -1:
            evidence = source[idx:idx + len(value)]
            return (0.99, evidence, (idx, idx + len(value)))
        return (0.0, None, None)
    
    def _numeric_match(self, source: str, value: Any) -> Tuple[float, Optional[str]]:
        """Find numeric values in source text."""
        if isinstance(value, str):
            clean_value = re.sub(r'[,$]', '', value)
            try:
                numeric_value = float(clean_value)
            except ValueError:
                return (0.0, None)
        else:
            numeric_value = float(value)
        number_pattern = r'[\$]?[\d,]+\.?\d*'
        matches = re.finditer(number_pattern, source)
        for match in matches:
            match_str = match.group()
            clean_match = re.sub(r'[,$]', '', match_str)
            try:
                source_num = float(clean_match)
                if abs(source_num - numeric_value) < 0.01:
                    return (0.95, match_str)
                if numeric_value != 0 and abs(source_num - numeric_value) / abs(numeric_value) < 0.01:
                    return (0.90, match_str)
            except ValueError:
                continue
        return (0.0, None)
    
    def _fuzzy_match(self, source: str, value: str) -> Tuple[float, Optional[str]]:
        """Fuzzy string matching using rapidfuzz."""
        if not RAPIDFUZZ_AVAILABLE:
            return (0.0, None)
        words = source.split()
        value_word_count = len(value.split())
        best_ratio = 0.0
        best_match = None
        for i in range(len(words)):
            for window_size in range(1, min(value_word_count + 3, len(words) - i + 1)):
                window = " ".join(words[i:i + window_size])
                ratio = fuzz.ratio(value.lower(), window.lower()) / 100.0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = window
        token_ratio = fuzz.token_sort_ratio(value.lower(), source.lower()) / 100.0
        if token_ratio > best_ratio:
            best_ratio = token_ratio
            best_match = f"[token match: {value}]"
        return (best_ratio, best_match)
    
    def _semantic_match(self, source: str, value: str) -> Tuple[float, Optional[str]]:
        """Semantic similarity using sentence embeddings."""
        if not EMBEDDINGS_AVAILABLE or self._embedding_model is None:
            return (0.0, None)
        sentences = re.split(r'[.!?\n]', source)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return (0.0, None)
        value_embedding = self._embedding_model.encode([value])[0]
        sentence_embeddings = self._embedding_model.encode(sentences)
        best_similarity = 0.0
        best_sentence = None
        for i, sent_emb in enumerate(sentence_embeddings):
            similarity = np.dot(value_embedding, sent_emb) / (np.linalg.norm(value_embedding) * np.linalg.norm(sent_emb))
            if similarity > best_similarity:
                best_similarity = similarity
                best_sentence = sentences[i]
        return (float(best_similarity), best_sentence)
    
    def _verify_complex_field(self, source_text: str, field_name: str, value: Union[list, dict], threshold: float) -> FieldResult:
        """Verify complex nested structures."""
        if isinstance(value, list):
            confidences = []
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    for k, v in item.items():
                        result = self._verify_field(source_text, f"{field_name}[{i}].{k}", v, threshold)
                        confidences.append(result.confidence)
                else:
                    result = self._verify_field(source_text, f"{field_name}[{i}]", item, threshold)
                    confidences.append(result.confidence)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return FieldResult(field_name=field_name, value=value, confidence=avg_confidence, method=VerificationMethod.FUZZY_MATCH, flagged=avg_confidence < threshold, reason=f"List with {len(value)} items, avg confidence: {avg_confidence:.2f}")
        elif isinstance(value, dict):
            confidences = []
            for k, v in value.items():
                result = self._verify_field(source_text, f"{field_name}.{k}", v, threshold)
                confidences.append(result.confidence)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return FieldResult(field_name=field_name, value=value, confidence=avg_confidence, method=VerificationMethod.FUZZY_MATCH, flagged=avg_confidence < threshold, reason=f"Dict with {len(value)} keys, avg confidence: {avg_confidence:.2f}")
        return FieldResult(field_name=field_name, value=value, confidence=0.0, method=VerificationMethod.NOT_FOUND, flagged=True, reason="Unknown complex type")
    
    @staticmethod
    def _looks_like_number(s: str) -> bool:
        """Check if string looks like a number."""
        clean = re.sub(r'[,$\s]', '', s)
        try:
            float(clean)
            return True
        except ValueError:
            return False


def grounding_validator(source_text: str, threshold: float = 0.7, groundcheck: Optional[GroundCheck] = None) -> Callable[[Any], Any]:
    """Pydantic validator that checks if a field value is grounded in source text."""
    gc = groundcheck or GroundCheck(confidence_threshold=threshold)
    def validator(value: Any) -> Any:
        result = gc._verify_field(source_text, "field", value, threshold)
        if result.flagged:
            raise ValueError(f"Value '{value}' not grounded in source text. Confidence: {result.confidence:.2f}, Method: {result.method.value}")
        return value
    return validator


def verify_extraction(source_text: str, extracted_data: Dict[str, Any], threshold: float = 0.7, raise_on_hallucination: bool = False) -> VerificationResult:
    """Convenience function to verify extracted data against source text."""
    gc = GroundCheck(confidence_threshold=threshold)
    result = gc.verify(source_text, extracted_data)
    if raise_on_hallucination and result.flagged_fields:
        raise ValueError(f"Hallucination detected in fields: {result.flagged_fields}. Overall confidence: {result.overall_confidence:.2f}")
    return result


__all__ = ["GroundCheck", "VerificationResult", "FieldResult", "VerificationMethod", "grounding_validator", "verify_extraction"]
