---
title: Using IBM watsonx.ai for Inference
description: Learn how to use IBM watsonx.ai and LiteLLM for structured outputs, including setup, installation, and coding examples.
---

# Structured Outputs with IBM watsonx.ai

You can use IBM watsonx.ai for inference using [LiteLLM](https://docs.litellm.ai/docs/providers/watsonx).

## Prerequisites

- IBM Cloud Account
- API Key from IBM Cloud IAM: https://cloud.ibm.com/iam/apikeys
- Project ID (from watsonx.ai instance URL: https://dataplatform.cloud.ibm.com/projects/<WATSONX_PROJECT_ID>/)

## Install

```bash
uv add instructor[litellm]
```

## Example

```python
import os

import litellm
from litellm import completion
from pydantic import BaseModel, Field

import instructor
from instructor import Mode

litellm.drop_params = True  # watsonx.ai doesn't support `json_mode`

os.environ["WATSONX_URL"] = "https://us-south.ml.cloud.ibm.com"
os.environ["WATSONX_API_KEY"] = ""
os.environ["WATSONX_PROJECT_ID"] = ""
# Additional options: https://docs.litellm.ai/docs/providers/watsonx


class Company(BaseModel):
    name: str = Field(description="name of the company")
    year_founded: int = Field(description="year the company was founded")


client = instructor.from_litellm(completion, mode=Mode.JSON)

resp = client.chat.completions.create(
    model="watsonx/meta-llama/llama-3-8b-instruct",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": """\
Given the following text, create a Company object:

IBM was founded in 1911 as the Computing-Tabulating-Recording Company (CTR), a holding company of manufacturers of record-keeping and measuring systems.
""",
        }
    ],
    project_id=os.environ["WATSONX_PROJECT_ID"],
    response_model=Company,
)

print(resp.model_dump_json(indent=2))
"""
{
  "name": "IBM",
  "year_founded": 1911
}
"""
```
