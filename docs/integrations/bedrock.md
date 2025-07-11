---
title: Structured Outputs with AWS Bedrock and Pydantic
description: Learn how to use AWS Bedrock with Instructor for structured JSON outputs using Pydantic models. Create type-safe, validated responses from AWS Bedrock LLMs with Python.
---

# Structured Outputs with AWS Bedrock

This guide demonstrates how to use AWS Bedrock with Instructor to generate structured outputs. You'll learn how to use AWS Bedrock's LLM models with Pydantic to create type-safe, validated responses.

## Prerequisites

You'll need to have an AWS account with access to Bedrock and the appropriate permissions. You'll also need to set up your AWS credentials.

```bash
pip install "instructor[bedrock]"
```

## AWS Bedrock

AWS Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon through a single API.

## Auto Client Setup

For simplified setup, you can use the auto client pattern:

```python
import instructor

# Auto client with model specification
client = instructor.from_provider("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

# The auto client automatically handles:
# - AWS credential detection from environment
# - Region configuration (defaults to us-east-1)
# - Mode selection based on model (Claude models use BEDROCK_TOOLS)
```

## Deprecation Notice

> **Deprecation Notice:**
>
> The `_async` argument to `instructor.from_bedrock` is deprecated. Please use `async_client=True` for async clients instead. Support for `_async` may be removed in a future release. All new code and examples should use `async_client`.

### Environment Configuration

Set your AWS credentials and region:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

Or configure using AWS CLI:

```bash
aws configure
```

## Sync Example

```python
import boto3
import instructor
from pydantic import BaseModel

bedrock_client = boto3.client('bedrock-runtime')
client = instructor.from_bedrock(bedrock_client)

class User(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)
# > User(name='Jason', age=25)
```

## Async Example

> **Warning:**
> AWS Bedrock's official SDK (`boto3`) does not support async natively. If you need to call Bedrock from async code, you can use `asyncio.to_thread` to run synchronous Bedrock calls in a non-blocking way.

```python
import instructor
from pydantic import BaseModel
import asyncio

client = instructor.from_provider("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")

class User(BaseModel):
    name: str
    age: int

def get_user():
    return client.chat.completions.create(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "Extract Jason is 25 years old"}],
        response_model=User,
    )

async def get_user_async():
    return await asyncio.to_thread(get_user)

user = asyncio.run(get_user_async())
print(user)
```

## Supported Modes

AWS Bedrock supports the following modes with Instructor:

- `BEDROCK_TOOLS`: Uses function calling for models that support it (like Claude models)
- `BEDROCK_JSON`: Direct JSON response generation

```python
import boto3
import instructor
from instructor import Mode
from pydantic import BaseModel

bedrock_client = boto3.client('bedrock-runtime')
client = instructor.from_bedrock(bedrock_client, mode=Mode.BEDROCK_TOOLS)

class User(BaseModel):
    name: str
    age: int
```

## OpenAI Compatibility: Flexible Input Format and Model Parameter

Instructor’s Bedrock integration supports both OpenAI-style and Bedrock-native message formats, as well as any mix of the two. You can use either:

- **OpenAI-style**:  
  `{"role": "user", "content": "Extract: Jason is 25 years old"}`

- **Bedrock-native**:  
  `{"role": "user", "content": [{"text": "Extract: Jason is 25 years old"}]}`

- **Mixed**:  
  You can freely mix OpenAI-style and Bedrock-native messages in the same request. The integration will automatically convert OpenAI-style messages to the correct Bedrock format, while preserving any Bedrock-native fields you provide.

This flexibility also applies to other keyword arguments, such as the model name:

- You can use either `model` (OpenAI-style) or `modelId` (Bedrock-native) as a keyword argument.  
- If you provide `model`, Instructor will automatically convert it to `modelId` for Bedrock.
- If you provide both, `modelId` takes precedence.

**Example:**

```python
import instructor

messages = [
    {"role": "system", "content": "Extract the name and age."},  # OpenAI-style
    {"role": "user", "content": [{"text": "Extract: Jason is 25 years old"}]},  # Bedrock-native
    {"role": "assistant", "content": "Sure! Jason is 25."},  # OpenAI-style
]

# Both of these are valid:
user = client.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",  # OpenAI-style
    messages=messages,
    response_model=User,
)

user = client.create(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",  # Bedrock-native
    messages=messages,
    response_model=User,
)
```

All of the above will work seamlessly with Instructor’s Bedrock integration.

## Nested Objects

```python
import boto3
import instructor
from pydantic import BaseModel

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock-runtime')

# Enable instructor patches for Bedrock client
client = instructor.from_bedrock(bedrock_client)


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


# Create structured output with nested objects
user = client.chat.completions.create(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[
        {
            "role": "user",
            "content": """
            Extract: Jason is 25 years old.
            He lives at 123 Main St, New York, USA
            and has a summer house at 456 Beach Rd, Miami, USA
        """,
        },
    ],
    response_model=User,
)

print(user)
#> User(
#>     name='Jason',
#>     age=25,
#>     addresses=[
#>         Address(street='123 Main St', city='New York', country='USA'),
#>         Address(street='456 Beach Rd', city='Miami', country='USA')
#>     ]
#> )
```

## Modern Models and Features

### Latest Model Support

AWS Bedrock supports many modern foundation models:

```python
import instructor

# Claude 3.5 models (latest)
client = instructor.from_provider("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
# or
client = instructor.from_provider("bedrock/anthropic.claude-3-5-haiku-20241022-v1:0")

# Amazon Nova models (multimodal)
client = instructor.from_provider("bedrock/amazon.nova-micro-v1:0")

# Meta Llama 3 models
client = instructor.from_provider("bedrock/meta.llama3-70b-instruct-v1:0")

# Mistral models
client = instructor.from_provider("bedrock/mistral.mistral-large-2402-v1:0")
```

### Advanced Configuration

```python
import boto3
import instructor

# Custom AWS configuration
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='us-west-2',
    aws_access_key_id='your_key',
    aws_secret_access_key='your_secret'
)

client = instructor.from_bedrock(
    bedrock_client,
    mode=instructor.Mode.BEDROCK_TOOLS
)

# Advanced inference configuration
user = client.chat.completions.create(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "Extract user info"}],
    response_model=User,
    inferenceConfig={
        "maxTokens": 2048,
        "temperature": 0.1,
        "topP": 0.9,
        "stopSequences": ["STOP"]
    }
)
```