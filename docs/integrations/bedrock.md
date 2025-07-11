---
title: Structured Outputs with AWS Bedrock and Pydantic
description: Learn how to use AWS Bedrock with Instructor for structured JSON outputs using Pydantic models. Create type-safe, validated responses from AWS Bedrock LLMs with Python.
---

# Structured Outputs with AWS Bedrock

This guide demonstrates how to use AWS Bedrock with Instructor to generate structured outputs. You'll learn how to use AWS Bedrock's LLM models with Pydantic to create type-safe, validated responses.

## Flexible Input Format and Model Parameter

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

## Prerequisites

You'll need to have an AWS account with access to Bedrock and the appropriate permissions. You'll also need to set up your AWS credentials.

```bash
pip install "instructor[bedrock]"
```

## AWS Bedrock

AWS Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon through a single API.

### Sync Example

```python
import boto3
import instructor
from pydantic import BaseModel

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock-runtime')

# Enable instructor patches for Bedrock client
client = instructor.from_provider("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")


class User(BaseModel):
    name: str
    age: int


# Create structured output
user = client.chat.completions.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[
        {"role": "system", "content": "Extract the name and age from the user's message."},
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)
# > User(name='Jason', age=25)
```

### Async Example

```python
import boto3
import instructor
from pydantic import BaseModel
import asyncio

async_client = instructor.from_provider(
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    async_client=True,
)

class User(BaseModel):
    name: str
    age: int

async def get_user_async():
    return await async_client.converse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[
            {"role": "system", "content": "Extract the name and age from the user's message."},
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
    )

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

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock-runtime')

# Enable instructor patches for Bedrock client with specific mode
client = instructor.from_provider("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")


class User(BaseModel):
    name: str
    age: int


# Create structured output
user = client.converse(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)
# > User(name='Jason', age=25)
```

## Nested Objects

```python
import boto3
import instructor
from pydantic import BaseModel

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock-runtime')

# Enable instructor patches for Bedrock client
client = instructor.from_provider("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


# Create structured output with nested objects
user = client.converse(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
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

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Boto3 Bedrock Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock.html)
