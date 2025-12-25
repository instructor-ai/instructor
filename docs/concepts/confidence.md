---
title: Confidence Scoring
description: Get TRUE confidence scores from LLM extractions using token logprobs.
---

# Confidence Scoring

Confidence scoring provides real probability scores for LLM extractions by analyzing token log probabilities. Zero extra API calls required.

## The Problem

LLMs don't tell you when they're guessing. A confident-sounding answer might be completely uncertain internally.
```python
# LLM returns this with equal confidence in its tone:
{"name": "John Smith"}      # Actually 99% certain
{"email": "john@fake.com"}  # Actually 45% certain - likely hallucinated!
```

## The Solution

Confidence scoring uses **token logprobs** - the actual probabilities the model assigned to each token:
```python
from instructor import score_confidence

# Get confidence from existing response
confidence = score_confidence(response, extracted_data)

print(confidence.overall)           # 0.87
print(confidence.is_reliable)       # True
print(confidence.low_confidence_fields)  # ["email"]
```

## How It Works
```
LLM generates: {"name": "John"}

Token probabilities:
  "{" → 99.9%   (very confident)
  "name" → 98.5% (very confident)  
  "John" → 94.2% (confident)
  "}" → 99.1%   (very confident)

Overall confidence: 97.9%
```

When the model is uncertain:
```
Token probabilities:
  "email" → 72.3%  (somewhat confident)
  "john@" → 45.1%  (uncertain - guessing!)
  
Field confidence: 45.1%  ← Flagged as LOW
```

## Quick Start

### With OpenAI Directly
```python
from openai import OpenAI
from instructor import score_confidence
import json

client = OpenAI()

# Enable logprobs in your request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Extract: John Smith, john@example.com"}],
    response_format={"type": "json_object"},
    logprobs=True,  # Required for confidence scoring
)

data = json.loads(response.choices[0].message.content)
confidence = score_confidence(response, data, model="gpt-4o-mini")

print(f"Overall: {confidence.overall:.1%}")
for field, fc in confidence.fields.items():
    print(f"  {field}: {fc.confidence:.1%} ({fc.level.value})")
```

### Enable Logprobs Helper
```python
from instructor import enable_logprobs

kwargs = {"model": "gpt-4o-mini", "messages": [...]}
kwargs = enable_logprobs(kwargs)  # Adds logprobs=True
```

## API Reference

### ConfidenceScorer

Main class for scoring confidence:
```python
from instructor import ConfidenceScorer

scorer = ConfidenceScorer(
    high_threshold=0.90,    # >= 90% = HIGH
    medium_threshold=0.75,  # >= 75% = MEDIUM  
    low_threshold=0.50,     # >= 50% = LOW, else VERY_LOW
)

result = scorer.score(response, extracted_data, model="gpt-4o-mini")
```

### ConfidenceResult

Result object with all confidence data:

| Property | Type | Description |
|----------|------|-------------|
| `overall` | float | Overall confidence (0.0-1.0) |
| `level` | ConfidenceLevel | HIGH/MEDIUM/LOW/VERY_LOW |
| `is_reliable` | bool | True if level is HIGH |
| `fields` | dict | Field-level confidence |
| `low_confidence_fields` | list | Fields with LOW/VERY_LOW |
| `token_count` | int | Tokens analyzed |
| `processing_time_ms` | float | Processing time |

### ConfidenceLevel
```python
from instructor import ConfidenceLevel

ConfidenceLevel.HIGH      # >= 90%
ConfidenceLevel.MEDIUM    # >= 75%
ConfidenceLevel.LOW       # >= 50%
ConfidenceLevel.VERY_LOW  # < 50%
```

### LowConfidenceError

Exception for enforcing confidence thresholds:
```python
from instructor import LowConfidenceError

if confidence.overall < 0.80:
    raise LowConfidenceError(confidence, threshold=0.80)
```

## Performance

| Metric | Value |
|--------|-------|
| Extra API calls | **0** |
| Processing time | **< 1ms** |
| Dependencies | **None** |
| Memory overhead | **Minimal** |

## Combining with GroundCheck

Use both for maximum reliability:
```python
from instructor import score_confidence, verify_extraction

# 1. Get confidence (model certainty)
confidence = score_confidence(response, data)

# 2. Verify grounding (factual accuracy)  
grounding = verify_extraction(source_text, data)

# 3. Combined reliability check
is_reliable = (
    confidence.overall >= 0.85 and
    grounding.is_reliable
)
```

| Method | What It Measures |
|--------|------------------|
| **Confidence** | "How sure was the model?" |
| **GroundCheck** | "Is the value in the source?" |

## Best Practices

1. **Always enable logprobs** - Set `logprobs=True` in API calls
2. **Check low_confidence_fields** - These need verification
3. **Set appropriate thresholds** - Higher for critical data
4. **Combine with GroundCheck** - Maximum reliability
5. **Log confidence scores** - Track extraction quality over time
