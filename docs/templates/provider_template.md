---
title: [Provider Name]
description: Guide to using instructor with [Provider Name]
---

# Structured outputs with [Provider Name], a complete guide w/ instructor

[Brief introduction to the provider, what models they offer, and why someone would use them]

## Quick Start

First, install the required packages:

```bash
pip install "instructor[provider-specific-extras]"
```

You'll need to set up authentication:

```bash
export PROVIDER_API_KEY=your_api_key_here
# Add any other environment variables needed
```

## Basic Example

Here's how to extract structured data using [Provider Name]:

```python
import instructor
from provider_sdk import ClientClass
from pydantic import BaseModel

# Initialize the client
client = instructor.from_provider(
    ClientClass(
        api_key="your_api_key_here",
        # Other configuration options
    ),
    mode=instructor.Mode.PROVIDER_SPECIFIC_MODE,
)

# Define your data structure
class UserExtract(BaseModel):
    name: str
    age: int

# Extract structured data
user = client.chat.completions.create(
    model="provider-model-name",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(user)
# Expected output: UserExtract(name='Jason', age=25)
```

## Async Example

For asynchronous use cases:

```python
import instructor
import asyncio
from provider_sdk import AsyncClientClass
from pydantic import BaseModel

# Initialize the async client
client = instructor.from_provider(AsyncClientClass())

class UserExtract(BaseModel):
    name: str
    age: int

async def extract_data():
    user = await client.chat.completions.create(
        model="provider-model-name",
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    return user

# Run the async function
user = asyncio.run(extract_data())
print(user)
```

## Supported Modes

[Provider Name] supports the following instructor modes:

- `Mode.MODE_1` - Description of when to use this mode
- `Mode.MODE_2` - Description of when to use this mode
- [Additional modes as needed]

## Streaming Support

You can stream results with [Provider Name]:

```python
# Streaming partial results example code
```

## Provider-Specific Features

[Describe any special features or considerations specific to this provider]

## Models

[Provider Name] offers the following models:

- `model-1` - Description of capabilities
- `model-2` - Description of capabilities
- [More models as appropriate]