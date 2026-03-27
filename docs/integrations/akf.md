# AKF Integration — Trust Metadata for Structured LLM Outputs

## Overview

[AKF](https://github.com/HMAKT99/AKF) (Agent Knowledge Format) is the AI native file format — EXIF for AI. It embeds trust scores, source provenance, and compliance metadata into LLM outputs.

When Instructor extracts structured data from LLMs, AKF can stamp the output with:
- **Trust score** based on the model and source tier
- **Source provenance** — which model, what prompt, what evidence
- **Compliance metadata** — EU AI Act Article 50 readiness

## Example: Stamp Instructor Output

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI
from akf import stamp

class UserInfo(BaseModel):
    name: str
    age: int

client = instructor.from_openai(OpenAI())
user = client.chat.completions.create(
    model="gpt-4",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John is 30 years old"}]
)

# Stamp the output with AKF trust metadata
akf_unit = stamp(
    content=f"Extracted: {user.model_dump_json()}",
    confidence=0.85,
    agent="instructor",
    model="gpt-4",
    evidence=["structured extraction via instructor"],
)

# Save with trust metadata
import json
with open("output.json", "w") as f:
    json.dump({"data": user.model_dump(), "_akf": akf_unit.to_dict()}, f)
```

## Why

- 47% of developers distrust AI-generated outputs (2026)
- Structured extraction looks authoritative but may contain hallucinations
- AKF trust scores differentiate verified vs unverified structured data
- EU AI Act Article 50 (Aug 2, 2026) requires transparency metadata

## Install

```bash
pip install akf
```

## Links

- [akf.dev](https://akf.dev)
- [GitHub](https://github.com/HMAKT99/AKF)
