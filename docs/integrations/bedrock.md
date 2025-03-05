---
title: "Structured outputs with AWS Bedrock, a complete guide w/ instructor"
description: Learn how to combine AWS Bedrock and Instructor clients to create structured outputs with complex properties in Python.
---

# Structured outputs with AWS Bedrock, a complete guide w/ instructor

[AWS Bedrock](https://aws.amazon.com/bedrock/) is Amazon's fully managed service for foundation models. With Instructor, you can easily extract structured data from Bedrock models.

Let's first install the instructor client with AWS Bedrock support:

```bash
pip install "instructor[bedrock]"
```

Once we've done so, getting started is as simple as using our `from_bedrock` method to patch the client.

## Basic Usage

```python
# Standard library imports
import os
from typing import List

# Third-party imports
import boto3
import instructor
from pydantic import BaseModel, Field

# Define your models with proper type annotations
class Properties(BaseModel):
    """Model representing a key-value property."""
    name: str = Field(description="The name of the property")
    value: str = Field(description="The value of the property")


class User(BaseModel):
    """Model representing a user with properties."""
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    properties: List[Properties] = Field(description="List of user properties")

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock-runtime')

# Initialize the instructor client
client = instructor.from_bedrock(
    client=bedrock_client,
    mode=instructor.Mode.BEDROCK_JSON  # Use JSON mode
)

try:
    # Extract structured data
    user_response = client.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",  # Use appropriate model ID
        messages=[
            {
                "role": "system",
                "content": "Extract structured information based on the user's request."
            },
            {
                "role": "user",
                "content": "Create a user for a model with a name, age, and properties.",
            }
        ],
        response_model=User,
    )
    
    # Print the result as formatted JSON
    print(user_response.model_dump_json(indent=2))
    # {
    #   "name": "John Doe",
    #   "age": 35,
    #   "properties": [
    #     {
    #       "name": "City",
    #       "value": "New York"
    #     },
    #     {
    #       "name": "Occupation",
    #       "value": "Software Engineer"
    #     }
    #   ]
    # }
except instructor.exceptions.InstructorError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Instructor Modes

We provide multiple modes to work with the different response formats that AWS Bedrock supports:

1. `instructor.Mode.BEDROCK_JSON`: Uses AWS Bedrock's JSON response capability to extract the desired response model
2. `instructor.Mode.BEDROCK_TOOLS`: Uses AWS Bedrock's tool calling capabilities (when available with supported models)

In general, we recommend using `Mode.BEDROCK_JSON` for most use cases, but `Mode.BEDROCK_TOOLS` may provide better results for complex schemas when using models that support it.

## Async Support

Instructor also supports asynchronous operations with AWS Bedrock:

```python
import asyncio
import boto3
import instructor
from pydantic import BaseModel, Field

class Response(BaseModel):
    summary: str = Field(description="A concise summary of the text")
    keywords: list[str] = Field(description="Key topics mentioned in the text")

async def main():
    # Initialize the Bedrock client
    bedrock_client = boto3.client('bedrock-runtime')
    
    # Initialize the async instructor client
    client = instructor.from_bedrock(
        client=bedrock_client,
        mode=instructor.Mode.BEDROCK_JSON,
        async_client=True  # Enable async support
    )
    
    # Make an async request
    response = await client.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[
            {"role": "user", "content": "Summarize the benefits of cloud computing."}
        ],
        response_model=Response
    )
    
    print(f"Summary: {response.summary}")
    print(f"Keywords: {', '.join(response.keywords)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuring AWS Credentials

The simplest way to set up authentication for AWS Bedrock is to configure your AWS credentials using the AWS CLI:

```bash
aws configure
```

Alternatively, you can provide credentials in your code:

```python
import boto3
import instructor

# Create Bedrock client with explicit credentials
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='us-west-2',
    aws_access_key_id='YOUR_ACCESS_KEY',
    aws_secret_access_key='YOUR_SECRET_KEY'
)

# Use the client with instructor
client = instructor.from_bedrock(bedrock_client)
```

## Supported Models

AWS Bedrock provides access to various foundation models. When using instructor with Bedrock, you'll need to specify the appropriate model ID:

- Anthropic Claude models: `anthropic.claude-3-sonnet-20240229-v1:0`, `anthropic.claude-3-haiku-20240307-v1:0`, etc.
- Amazon Titan models: `amazon.titan-text-express-v1`, etc.
- AI21 Jurassic models: `ai21.j2-ultra-v1`, etc.
- Cohere models: `cohere.command-text-v14`, etc.

Refer to the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html) for a complete list of available models and their capabilities.

## Error Handling

When working with AWS Bedrock, you might encounter various errors. Here's how to handle common issues:

```python
import botocore.exceptions
import instructor.exceptions

try:
    response = client.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "Extract data..."}],
        response_model=YourModel
    )
    # Process response...
except botocore.exceptions.ClientError as e:
    # Handle AWS-specific errors
    error_code = e.response['Error']['Code']
    if error_code == 'AccessDeniedException':
        print("Check your AWS permissions for Bedrock access")
    elif error_code == 'ValidationException':
        print("Invalid request parameters")
    elif error_code == 'ThrottlingException':
        print("Rate limit exceeded")
    else:
        print(f"AWS error: {error_code}")
except instructor.exceptions.InstructorError as e:
    # Handle validation errors
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```