"""Tests for confidence scoring module."""

import pytest
import math
from unittest.mock import MagicMock
from instructor.confidence import (
    ConfidenceScorer,
    ConfidenceResult,
    FieldConfidence,
    ConfidenceLevel,
    score_confidence,
    enable_logprobs,
    LowConfidenceError,
)


class TestConfidenceScorer:
    """Test ConfidenceScorer class."""
    
    def test_logprob_to_probability(self):
        scorer = ConfidenceScorer()
        assert scorer.logprob_to_probability(0) == 1.0
        prob = scorer.logprob_to_probability(-1)
        assert 0.36 < prob < 0.37
    
    def test_get_confidence_level(self):
        scorer = ConfidenceScorer()
        assert scorer.get_confidence_level(0.95) == ConfidenceLevel.HIGH
        assert scorer.get_confidence_level(0.85) == ConfidenceLevel.MEDIUM
        assert scorer.get_confidence_level(0.60) == ConfidenceLevel.LOW
        assert scorer.get_confidence_level(0.30) == ConfidenceLevel.VERY_LOW
    
    def test_custom_thresholds(self):
        scorer = ConfidenceScorer(high_threshold=0.95, medium_threshold=0.80, low_threshold=0.60)
        assert scorer.get_confidence_level(0.92) == ConfidenceLevel.MEDIUM
        assert scorer.get_confidence_level(0.75) == ConfidenceLevel.LOW


class TestConfidenceResult:
    """Test ConfidenceResult class."""
    
    def test_is_reliable(self):
        result = ConfidenceResult(
            overall=0.95, level=ConfidenceLevel.HIGH, fields={},
            token_count=10, processing_time_ms=0.5
        )
        assert result.is_reliable is True
        
        result2 = ConfidenceResult(
            overall=0.70, level=ConfidenceLevel.MEDIUM, fields={},
            token_count=10, processing_time_ms=0.5
        )
        assert result2.is_reliable is False
    
    def test_low_confidence_fields(self):
        result = ConfidenceResult(
            overall=0.75, level=ConfidenceLevel.MEDIUM,
            fields={
                "name": FieldConfidence("name", "John", 0.95, ConfidenceLevel.HIGH),
                "email": FieldConfidence("email", "john@test.com", 0.45, ConfidenceLevel.VERY_LOW),
                "age": FieldConfidence("age", 30, 0.55, ConfidenceLevel.LOW),
            },
            token_count=20, processing_time_ms=0.5
        )
        low_fields = result.low_confidence_fields
        assert "email" in low_fields
        assert "age" in low_fields
        assert "name" not in low_fields
    
    def test_to_dict(self):
        result = ConfidenceResult(
            overall=0.85, level=ConfidenceLevel.MEDIUM,
            fields={"name": FieldConfidence("name", "John", 0.90, ConfidenceLevel.HIGH)},
            token_count=10, processing_time_ms=0.5, model="gpt-4o-mini"
        )
        d = result.to_dict()
        assert d["overall"] == 0.85
        assert d["level"] == "medium"
        assert d["model"] == "gpt-4o-mini"
        assert "name" in d["fields"]


class TestFieldConfidence:
    """Test FieldConfidence class."""
    
    def test_auto_level_assignment(self):
        fc = FieldConfidence("test", "value", 0.95, ConfidenceLevel.LOW)
        assert fc.level == ConfidenceLevel.HIGH
        
        fc2 = FieldConfidence("test", "value", 0.40, ConfidenceLevel.HIGH)
        assert fc2.level == ConfidenceLevel.VERY_LOW


class TestMockLLMResponse:
    """Test with mocked LLM responses."""
    
    def create_mock_response(self, tokens_with_logprobs):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_logprobs = MagicMock()
        
        mock_tokens = []
        for token, logprob in tokens_with_logprobs:
            mock_token = MagicMock()
            mock_token.token = token
            mock_token.logprob = logprob
            mock_tokens.append(mock_token)
        
        mock_logprobs.content = mock_tokens
        mock_choice.logprobs = mock_logprobs
        mock_response.choices = [mock_choice]
        return mock_response
    
    def test_high_confidence_extraction(self):
        tokens = [('{"', -0.01), ('name', -0.02), ('":', -0.01), ('"John', -0.05), ('"}', -0.01)]
        response = self.create_mock_response(tokens)
        
        scorer = ConfidenceScorer()
        result = scorer.score(response, {"name": "John"}, model="gpt-4o-mini")
        
        assert result.overall > 0.85
        assert result.model == "gpt-4o-mini"
    
    def test_low_confidence_extraction(self):
        tokens = [('{"', -0.5), ('name', -2.0), ('":', -0.5), ('"Maybe', -3.0), ('"}', -0.5)]
        response = self.create_mock_response(tokens)
        
        scorer = ConfidenceScorer()
        result = scorer.score(response, {"name": "Maybe"})
        
        assert result.overall < 0.70


class TestEnableLogprobs:
    """Test logprobs enabling helper."""
    
    def test_enable_logprobs(self):
        kwargs = {"model": "gpt-4o-mini", "messages": []}
        result = enable_logprobs(kwargs)
        assert result["logprobs"] is True
        assert result["model"] == "gpt-4o-mini"


class TestLowConfidenceError:
    """Test LowConfidenceError exception."""
    
    def test_error_message(self):
        result = ConfidenceResult(
            overall=0.45, level=ConfidenceLevel.VERY_LOW,
            fields={"name": FieldConfidence("name", "John", 0.40, ConfidenceLevel.VERY_LOW)},
            token_count=5, processing_time_ms=0.5
        )
        error = LowConfidenceError(result, threshold=0.70)
        assert error.threshold == 0.70
        assert error.confidence == result


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_extracted_data(self):
        mock_response = MagicMock()
        mock_response.choices = []
        
        scorer = ConfidenceScorer()
        result = scorer.score(mock_response, {})
        
        assert result.overall == 0.0
        assert len(result.fields) == 0
    
    def test_no_logprobs_in_response(self):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.logprobs = None
        mock_response.choices = [mock_choice]
        
        scorer = ConfidenceScorer()
        result = scorer.score(mock_response, {"name": "John"})
        
        assert isinstance(result, ConfidenceResult)
        assert result.token_count == 0


class TestPerformance:
    """Test performance characteristics."""
    
    def test_processing_time(self):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_logprobs = MagicMock()
        
        mock_tokens = []
        for i in range(100):
            mock_token = MagicMock()
            mock_token.token = f"token{i}"
            mock_token.logprob = -0.1
            mock_tokens.append(mock_token)
        
        mock_logprobs.content = mock_tokens
        mock_choice.logprobs = mock_logprobs
        mock_response.choices = [mock_choice]
        
        scorer = ConfidenceScorer()
        result = scorer.score(mock_response, {"field": "test"})
        
        assert result.processing_time_ms < 10.0
