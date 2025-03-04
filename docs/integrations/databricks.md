---
title: Databricks
description: Guide to using instructor with Databricks models
---

# Structured outputs with Databricks, a complete guide w/ instructor

[Databricks](https://www.databricks.com/) provides an AI platform with access to various models. This guide shows how to use instructor with Databricks to get structured outputs.

## Quick Start

First, install the required packages:

```bash
pip install instructor
```

You'll need a Databricks API key and workspace URL which you can set as environment variables:

```bash
export DATABRICKS_API_KEY=your_api_key_here
export DATABRICKS_HOST=your_workspace_url
```

## Basic Example

Here's how to extract structured data from Databricks models:

```python
import os
import instructor
from openai import OpenAI
from pydantic import BaseModel

# Initialize the client with Databricks base URL
client = instructor.from_openai(
    OpenAI(
        base_url="https://your-databricks-workspace-url/serving-endpoints/your-endpoint-name/invocations",
        api_key=os.environ["DATABRICKS_API_KEY"],
    ),
    mode=instructor.Mode.TOOLS,
)

# Define your data structure
class UserExtract(BaseModel):
    name: str
    age: int

# Extract structured data
user = client.chat.completions.create(
    model="databricks-model", # Your model name in Databricks
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(user)
# Output: UserExtract(name='Jason', age=25)
```

## Supported Modes

Databricks supports the same modes as OpenAI:

- `Mode.TOOLS`
- `Mode.JSON`
- `Mode.FUNCTIONS`
- `Mode.PARALLEL_TOOLS`
- `Mode.MD_JSON`
- `Mode.TOOLS_STRICT`
- `Mode.JSON_O1`

## Models

Databricks provides access to various models depending on your setup, including:

- Foundation models hosted on Databricks
- Custom fine-tuned models
- Open source models deployed on Databricks

