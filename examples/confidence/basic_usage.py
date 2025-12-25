"""
Confidence Scoring: Basic Usage Examples

Demonstrates how to get TRUE confidence scores from LLM extractions
using token log probabilities.

Requirements:
    pip install openai instructor
    export OPENAI_API_KEY='your-key'
"""

import os
import json

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("=" * 60)
    print("DEMO MODE - Using mock data (no API key found)")
    print("Set OPENAI_API_KEY to test with real LLM")
    print("=" * 60)
    
    # Demo with mock data
    from instructor import ConfidenceScorer, ConfidenceLevel
    from unittest.mock import MagicMock
    
    def create_mock_response(tokens_with_logprobs):
        """Create mock OpenAI response with logprobs."""
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
    
    print("\n" + "=" * 60)
    print("Example 1: High Confidence Extraction")
    print("=" * 60)
    
    # High confidence tokens (logprobs close to 0)
    high_conf_tokens = [
        ('{"', -0.01),
        ('name', -0.02),
        ('":', -0.01),
        ('"', -0.01),
        ('John', -0.05),
        (' Smith', -0.08),
        ('"}', -0.01),
    ]
    
    response = create_mock_response(high_conf_tokens)
    extracted = {"name": "John Smith"}
    
    scorer = ConfidenceScorer()
    result = scorer.score(response, extracted, model="gpt-4o-mini")
    
    print(f"\nExtracted: {extracted}")
    print(f"Overall Confidence: {result.overall:.1%}")
    print(f"Level: {result.level.value}")
    print(f"Is Reliable: {result.is_reliable}")
    
    print("\n" + "=" * 60)
    print("Example 2: Low Confidence Extraction")
    print("=" * 60)
    
    # Low confidence tokens (logprobs far from 0)
    low_conf_tokens = [
        ('{"', -0.5),
        ('email', -1.5),
        ('":', -0.5),
        ('"', -0.8),
        ('maybe', -3.0),  # Very uncertain!
        ('@unknown', -2.5),  # Very uncertain!
        ('.com', -1.8),
        ('"}', -0.5),
    ]
    
    response = create_mock_response(low_conf_tokens)
    extracted = {"email": "maybe@unknown.com"}
    
    result = scorer.score(response, extracted)
    
    print(f"\nExtracted: {extracted}")
    print(f"Overall Confidence: {result.overall:.1%}")
    print(f"Level: {result.level.value}")
    print(f"Is Reliable: {result.is_reliable}")
    print(f"⚠️  Low confidence - likely hallucinated!")
    
    print("\n" + "=" * 60)
    print("Example 3: Mixed Confidence Fields")
    print("=" * 60)
    
    mixed_tokens = [
        ('{"', -0.01),
        ('name', -0.02),
        ('":', -0.01),
        ('"John"', -0.05),  # High confidence
        (',', -0.01),
        ('"age', -0.02),
        ('":', -0.01),
        ('25', -2.0),  # Low confidence - uncertain
        ('}', -0.01),
    ]
    
    response = create_mock_response(mixed_tokens)
    extracted = {"name": "John", "age": 25}
    
    result = scorer.score(response, extracted)
    
    print(f"\nExtracted: {extracted}")
    print(f"Overall Confidence: {result.overall:.1%}")
    print("\nField-level breakdown:")
    for field_name, fc in result.fields.items():
        emoji = "✅" if fc.level == ConfidenceLevel.HIGH else "⚠️" if fc.level == ConfidenceLevel.MEDIUM else "❌"
        print(f"  {emoji} {field_name}: {fc.confidence:.1%} ({fc.level.value})")
    
    print(f"\nLow confidence fields: {result.low_confidence_fields}")

else:
    # Real LLM example
    from openai import OpenAI
    from instructor import score_confidence, enable_logprobs
    
    client = OpenAI()
    
    print("\n" + "=" * 60)
    print("Real LLM Confidence Scoring")
    print("=" * 60)
    
    # Test 1: Clear extraction
    print("\n📝 Test 1: Clear Data")
    text1 = "Contact: John Smith, Email: john.smith@company.com, Phone: 555-1234"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract contact info as JSON with fields: name, email, phone"},
            {"role": "user", "content": text1}
        ],
        response_format={"type": "json_object"},
        logprobs=True,
    )
    
    data = json.loads(response.choices[0].message.content)
    confidence = score_confidence(response, data, model="gpt-4o-mini")
    
    print(f"   Source: {text1}")
    print(f"   Extracted: {data}")
    print(f"   Confidence: {confidence.overall:.1%} ({confidence.level.value})")
    
    # Test 2: Ambiguous extraction
    print("\n📝 Test 2: Ambiguous Data")
    text2 = "Someone named John or maybe James works at some tech company"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract contact info as JSON with fields: name, email, company"},
            {"role": "user", "content": text2}
        ],
        response_format={"type": "json_object"},
        logprobs=True,
    )
    
    data = json.loads(response.choices[0].message.content)
    confidence = score_confidence(response, data, model="gpt-4o-mini")
    
    print(f"   Source: {text2}")
    print(f"   Extracted: {data}")
    print(f"   Confidence: {confidence.overall:.1%} ({confidence.level.value})")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
✅ Confidence scoring uses REAL token probabilities
✅ Zero extra API calls - just enable logprobs=True
✅ Identifies uncertain extractions automatically
✅ Combine with GroundCheck for maximum reliability
    """)

print("\n✅ Examples completed!")
