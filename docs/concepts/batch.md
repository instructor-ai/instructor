---
title: Batch Processing with Instructor - Cost Optimization Guide
description: Process multiple LLM requests efficiently using batch processing for 50% cost savings. Learn file-based and in-memory batch processing with OpenAI, Anthropic, and Google.
---

# Batch Processing

Batch processing allows you to send multiple requests in a single operation, which is more cost-effective and efficient for processing large datasets. Instructor supports batch processing across multiple providers with a unified interface.

## Supported Providers

### OpenAI
- **Models**: gpt-4o, gpt-4.1-mini, gpt-4-turbo, etc.
- **Cost Savings**: 50% discount on batch requests
- **Format**: Uses OpenAI's batch API with JSON schema for structured outputs

### Anthropic
- **Models**: claude-3-5-sonnet, claude-3-opus, claude-3-haiku
- **Cost Savings**: 50% discount on batch requests
- **Format**: Uses Anthropic's Message Batches API with tool calling for structured outputs
- **Documentation**: [Anthropic Message Batches API](https://docs.anthropic.com/en/api/creating-message-batches)

### Google GenAI
- **Models**: gemini-2.5-flash, gemini-2.0-flash, gemini-pro
- **Cost Savings**: 50% discount on batch requests
- **Format**: Uses Google Cloud Vertex AI batch prediction API
- **Documentation**: [Google Cloud Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini)

## Basic Usage

The `BatchProcessor` provides a complete interface for batch processing including job submission, status monitoring, and result retrieval.

### File-based Batch Processing (Traditional)

```python
from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Create processor with model specification
processor = BatchProcessor("openai/gpt-4.1-mini", User)

# Prepare your message conversations
messages_list = [
    [
        {"role": "system", "content": "Extract user information from text."},
        {"role": "user", "content": "Hi, I'm Alice and I'm 28 years old."},
    ],
    [
        {"role": "system", "content": "Extract user information from text."},
        {"role": "user", "content": "Hello, I'm Bob, 35 years old."},
    ],
]

# Create batch file
processor.create_batch_from_messages(
    file_path="batch_requests.jsonl",  # Specify file path for disk-based processing
    messages_list=messages_list,
    max_tokens=200,
    temperature=0.1,
)

# Submit batch job to provider
batch_id = processor.submit_batch("batch_requests.jsonl")
print(f"Batch job submitted: {batch_id}")
#> Batch job submitted: batch_696fac8b9ec48190a59b5fc30a7f1f0b

# Check batch status
status = processor.get_batch_status(batch_id)
print(f"Status: {status['status']}")
#> Status: validating

# Retrieve results when completed
if status['status'] in ['completed', 'ended', 'JOB_STATE_SUCCEEDED']:
    from instructor.batch import filter_successful, filter_errors, extract_results

    all_results = processor.retrieve_results(batch_id)
    successful_results = filter_successful(all_results)
    error_results = filter_errors(all_results)
    extracted_users = extract_results(all_results)

    print(f"Successfully parsed: {len(successful_results)} results")
    print(f"Errors: {len(error_results)} results")

    # Access results with custom_id tracking
    for result in successful_results:
        print(f"ID: {result.custom_id}, User: {result.result}")

    # Or just get the extracted objects
    for user in extracted_users:
        print(f"Name: {user.name}, Age: {user.age}")
```

### In-Memory Batch Processing (Serverless-Friendly)

For serverless deployments and applications that prefer memory-only operations:

```python
import time
from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Create processor
processor = BatchProcessor("openai/gpt-5-nano", User)

# Prepare your message conversations
messages_list = [
    [
        {"role": "system", "content": "Extract user information from the text."},
        {
            "role": "user",
            "content": "John Doe is 25 years old and his email is john@example.com",
        },
    ],
    [
        {"role": "system", "content": "Extract user information from the text."},
        {
            "role": "user",
            "content": "Jane Smith, age 30, can be reached at jane.smith@company.com",
        },
    ],
]

# Create batch in memory (no file_path = in-memory mode)
batch_buffer = processor.create_batch_from_messages(
    messages_list,
    file_path=None,  # This enables in-memory mode
    max_tokens=150,
    temperature=0.1,
)

print(f"Created batch buffer: {type(batch_buffer)}")
#> Created batch buffer: <class '_io.BytesIO'>
print(f"Buffer size: {len(batch_buffer.getvalue())} bytes")
#> Buffer size: 1266 bytes

# Submit the batch using the in-memory buffer
batch_id = processor.submit_batch(
    batch_buffer, metadata={"description": "In-memory batch example"}
)

print(f"Batch submitted successfully! ID: {batch_id}")
#> Batch submitted successfully! ID: batch_...

# Poll for completion
while True:
    status = processor.get_batch_status(batch_id)
    current_status = status.get("status", "unknown")
    print(f"Status: {current_status}")

    if current_status in ["completed", "failed", "cancelled", "expired"]:
        break
    time.sleep(10)

# Retrieve results
if status.get("status") == "completed":
    results = processor.get_results(batch_id)
else:
    results = []

successful_results = [r for r in results if hasattr(r, "result")]
error_results = [r for r in results if hasattr(r, "error_message")]

print(f"Successful: {len(successful_results)}")
#> Successful: 2
print(f"Errors: {len(error_results)}")
#> Errors: 2

for result in successful_results:
    user = result.result
    print(f"- {user.name}, {user.age} years old")
    #> - John Doe, 25 years old
    #> - Jane Smith, 30 years old
```

## In-Memory vs File-Based Processing

### When to Use In-Memory Processing

**✅ Ideal for:**
- **Serverless deployments** (AWS Lambda, Google Cloud Functions, Azure Functions)
- **Containerized applications** where disk I/O should be minimized
- **Security-sensitive environments** where temporary files on disk are not desired
- **High-performance applications** that want to avoid file system overhead

**Key Benefits:**
- **No disk I/O** - Perfect for serverless environments with read-only file systems
- **Faster processing** - No file system overhead
- **Better security** - No temporary files left on disk
- **Cleaner code** - No file cleanup required
- **Memory efficient** - BytesIO buffers are automatically garbage collected

### When to Use File-Based Processing

**✅ Ideal for:**
- **Large batch jobs** where memory usage is a concern
- **Long-running applications** with persistent storage
- **Debugging scenarios** where you want to inspect the batch file
- **Audit requirements** where batch requests need to be saved

### Comparison Example

```python
from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


processor = BatchProcessor("openai/gpt-5-nano", User)
messages_list = [
    [{"role": "user", "content": "Extract: John, 25, john@example.com"}],
    [{"role": "user", "content": "Extract: Jane, 30, jane@example.com"}],
]

# File-based approach (traditional)
file_path = processor.create_batch_from_messages(
    messages_list,
    file_path="temp_batch.jsonl",  # Creates file on disk
)
batch_id = processor.submit_batch(file_path)
# Remember to clean up: os.remove(file_path)

# In-memory approach (new)
buffer = processor.create_batch_from_messages(
    messages_list,
    file_path=None,  # Returns BytesIO buffer
)
batch_id = processor.submit_batch(buffer)
# No cleanup required - buffer is garbage collected
```

### BytesIO Lifecycle Management

When using in-memory batch processing, the BytesIO buffer lifecycle is managed as follows:

1. **Creation**: The `create_batch_from_messages()` method creates and returns a BytesIO buffer
2. **Ownership**: The caller owns the buffer and is responsible for its lifecycle
3. **Submission**: The `submit_batch()` method reads from the buffer but doesn't close it
4. **Cleanup**: Python's garbage collector automatically cleans up the buffer when it goes out of scope

**Best Practices:**
- The buffer is automatically cleaned up when no longer referenced
- No explicit `.close()` call is needed for BytesIO objects
- If you need to reuse the buffer, call `.seek(0)` to reset position
- For very large batches, consider monitoring memory usage

```python
# Example: Reusing a buffer
from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


processor = BatchProcessor("openai/gpt-5-nano", User)
messages = [[{"role": "user", "content": "Extract: John, 25"}]]

buffer = processor.create_batch_from_messages(messages, file_path=None)

batch_id_1 = processor.submit_batch(buffer)

# Reset buffer position to reuse
buffer.seek(0)
batch_id_2 = processor.submit_batch(buffer)
```

## Provider-Specific Setup

### OpenAI Setup
```bash
export OPENAI_API_KEY="your-openai-key"
```

```python
from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Use OpenAI models
processor = BatchProcessor("openai/gpt-4.1-mini", User)
```

### Anthropic Setup
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

```python
from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Use Anthropic models
processor = BatchProcessor("anthropic/claude-3-5-sonnet-20241022", User)
```

### Google GenAI Setup

For Google GenAI batch processing, you need additional setup:

```bash
# Set up Google Cloud credentials
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"  # Optional, defaults to us-central1
export GCS_BUCKET="your-gcs-bucket-name"

# Authentication (choose one method):
# 1. Service account key file
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# 2. Or use gcloud auth for development
gcloud auth application-default login
```

**Required permissions:**
- `roles/aiplatform.user` for Vertex AI
- `roles/storage.objectUser` for Cloud Storage access

```python
from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Use Google GenAI models
_processor = BatchProcessor("google/gemini-2.5-flash", User)
```

## Maybe-Like Result Design

Instructor's batch API uses a Maybe/Result-like pattern where each result is either a success or error, with custom_id tracking:

```python
from instructor.batch import (
    BatchProcessor,
    filter_successful,
    filter_errors,
    extract_results,
    get_results_by_custom_id,
)

# Example: Retrieve results from a completed batch job
# Results are returned as a union type: BatchSuccess[T] | BatchError
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


processor = BatchProcessor("openai/gpt-4.1-mini", User)
batch_id = "batch_abc123"  # Example batch ID from your provider

try:
    all_results = processor.retrieve_results(batch_id)
except Exception:
    # Example: batch doesn't exist yet or hasn't completed
    all_results = []

# Filter and work with results
successful_results = filter_successful(all_results)  # List[BatchSuccess[T]]
error_results = filter_errors(all_results)  # List[BatchError]
extracted_objects = extract_results(all_results)  # List[T] (just the parsed objects)

# Access by custom_id (if results exist)
if all_results:
    results_by_id = get_results_by_custom_id(all_results)
    if "request-1" in results_by_id:
        user_result = results_by_id["request-1"]
        if user_result.success:
            print(f"Success: {user_result.result}")
        else:
            print(f"Error: {user_result.error_message}")
```

This design provides:
- **Type Safety**: Clear distinction between success and error cases
- **Custom ID Tracking**: Every result preserves its original custom_id
- **Functional Style**: Helper functions for filtering and extracting
- **Error Details**: Rich error information with error types and messages

## Processing Results

After your batch job completes, parse the results using the new Maybe-like API:

```python
from instructor.batch import (
    BatchProcessor,
    filter_successful,
    filter_errors,
    extract_results,
)

# Read results file (downloaded from provider)
try:
    with open("batch_results.jsonl") as f:
        results_content = f.read()

    # Create processor and parse results using the new Maybe-like pattern
    processor = BatchProcessor("openai/gpt-4.1-mini", None)
    all_results = processor.parse_results(results_content)
except FileNotFoundError:
    # Example: file doesn't exist yet
    all_results = []

# Filter results by type
successful_results = filter_successful(all_results)  # List[BatchSuccess[T]]
error_results = filter_errors(all_results)  # List[BatchError]
extracted_users = extract_results(all_results)  # List[T] (just the objects)

print(f"Successfully parsed: {len(successful_results)} results")
#> Successfully parsed: 0 results
print(f"Errors: {len(error_results)} results")
#> Errors: 0 results

# Access parsed data with custom_id tracking
for result in successful_results:
    print(f"ID: {result.custom_id}, User: {result.result}")

# Or access just the extracted objects
for user in extracted_users:
    print(f"Name: {user.name}, Age: {user.age}")

# Handle errors with details
for error in error_results:
    print(f"Error in {error.custom_id}: {error.error_message} ({error.error_type})")
```

## Provider Response Formats

### OpenAI Response Format
```json
{
  "custom_id": "request-0",
  "response": {
    "body": {
      "choices": [{
        "message": {
          "content": "{\"name\": \"Alice\", \"age\": 28}"
        }
      }]
    }
  }
}
```

### Anthropic Response Format
```json
{
  "custom_id": "request-0",
  "result": {
    "message": {
      "content": [{
        "type": "tool_use",
        "input": {"name": "Alice", "age": 28}
      }]
    }
  }
}
```

### Google GenAI Response Format
Based on [Google Cloud documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini#batch_prediction_inputs_and_outputs):

```json
{
  "response": {
    "candidates": [{
      "content": {
        "parts": [{
          "text": "{\"name\": \"Alice\", \"age\": 28}"
        }]
      }
    }]
  }
}
```

## Complete Workflow Example

Here's a complete example showing the full batch processing workflow:

```python
import os
import time
from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int
    occupation: str


def run_batch_workflow(provider_model: str):
    """Complete batch processing workflow"""

    # Sample conversations
    messages_list = [
        [
            {"role": "system", "content": "Extract user information from the text."},
            {
                "role": "user",
                "content": "Hi there! I'm Alice, 28 years old, working as a software engineer.",
            },
        ],
        [
            {"role": "system", "content": "Extract user information from the text."},
            {
                "role": "user",
                "content": "Hello! My name is Bob, I'm 35 and I work as a data scientist.",
            },
        ],
    ]

    # Step 1: Create processor
    processor = BatchProcessor(provider_model, User)

    # Step 2: Generate batch file
    batch_file = f"{provider_model.replace('/', '_')}_batch.jsonl"
    processor.create_batch_from_messages(
        file_path=batch_file,
        messages_list=messages_list,
        max_tokens=200,
        temperature=0.1,
    )
    print(f"✅ Created batch file: {batch_file}")
    #> ✅ Created batch file: openai_gpt-4.1-mini_batch.jsonl

    # Step 3: Submit batch job
    try:
        batch_id = processor.submit_batch(batch_file)
        #> 🚀 Batch job submitted: batch_696f82b224a481909b4a9250e061a085
        print(f"🚀 Batch job submitted: {batch_id}")
        #> 🚀 Batch job submitted: batch_696fac8e09f881908267992f2b9c9549

        # Step 4: Monitor batch status
        while True:
            #> 📊 Status: validating
            #> 📊 Status: in_progress
            #> 📊 Status: in_progress
            #> 📊 Status: completed
            status = processor.get_batch_status(batch_id)
            print(f"📊 Status: {status['status']}")
            #> 📊 Status: validating
            #> 📊 Status: in_progress
            #> 📊 Status: in_progress
            #> 📊 Status: in_progress
            #> 📊 Status: in_progress
            #> 📊 Status: in_progress
            #> 📊 Status: in_progress
            #> 📊 Status: completed

            # Check if completed (status varies by provider)
            if status['status'] in ['completed', 'ended', 'JOB_STATE_SUCCEEDED']:
                break
            elif status['status'] in ['failed', 'cancelled', 'JOB_STATE_FAILED']:
                print(f"❌ Batch job failed with status: {status['status']}")
                return

            # Wait before checking again
            time.sleep(30)

        # Step 5: Retrieve results using new Maybe-like API
        from instructor.batch import filter_successful, filter_errors, extract_results

        all_results = processor.retrieve_results(batch_id)
        successful_results = filter_successful(all_results)
        #> ✅ Successfully parsed: 2 results
        error_results = filter_errors(all_results)
        extracted_users = extract_results(all_results)

        print(f"✅ Successfully parsed: {len(successful_results)} results")
        #> ✅ Successfully parsed: 2 results
        if error_results:
            print(f"❌ Failed extractions: {len(error_results)}")
            # Show error details
            for error in error_results[:3]:  # Show first 3 errors
                print(f"   Error ({error.custom_id}): {error.error_message}")

        # Display results with custom_id tracking
        for result in successful_results:
            #>   - request-0: Alice, age 28, works as software engineer
            #>   - request-1: Bob, age 35, works as data scientist
            user = result.result
            print(
                f"  - {result.custom_id}: {user.name}, age {user.age}, works as {user.occupation}"
            )
            #>   - request-0: Alice, age 28, works as software engineer
            #>   - request-1: Bob, age 35, works as data scientist

    except Exception as e:
        print(f"❌ Error: {e}")
        """
        ❌ Error: Failed to retrieve Anthropic results: All 0 batch requests failed. No results will be available.
        """

    finally:
        # Cleanup
        if os.path.exists(batch_file):
            os.remove(batch_file)


# Usage with different providers
if __name__ == "__main__":
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        run_batch_workflow("openai/gpt-4.1-mini")

    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        run_batch_workflow("anthropic/claude-3-5-sonnet-20241022")

    # Google GenAI (requires additional setup)
    if os.getenv("GOOGLE_CLOUD_PROJECT") and os.getenv("GCS_BUCKET"):
        run_batch_workflow("google/gemini-2.5-flash")
```

## BatchProcessor API Reference

### Core Methods

- **`create_batch_from_messages(messages_list, file_path=None, max_tokens=1000, temperature=0.1)`**: 
  - Generate batch request file from message conversations
  - **Parameters:**
    - `messages_list`: List of message conversations
    - `file_path`: Path to save batch file. If `None`, returns BytesIO buffer (in-memory mode)
    - `max_tokens`: Maximum tokens per request
    - `temperature`: Temperature for generation
  - **Returns:** File path (str) or BytesIO buffer
  
- **`submit_batch(file_path_or_buffer, metadata=None, **kwargs)`**: 
  - Submit batch job to the provider and return job ID
  - **Parameters:**
    - `file_path_or_buffer`: File path (str) or BytesIO buffer
    - `metadata`: Optional metadata dict
  - **Returns:** Batch job ID (str)
  
- **`get_batch_status(batch_id)`**: Get current status of a batch job
- **`retrieve_results(batch_id)`**: Download and parse batch results
- **`parse_results(results_content)`**: Parse raw batch results into structured objects

## CLI Usage

Instructor also provides CLI commands for batch processing:

```bash
# List batch jobs
instructor batch list --model "openai/gpt-4.1-mini"

# Create batch from file
instructor batch create-from-file --file-path batch_requests.jsonl --model "openai/gpt-4.1-mini"

# Get batch results
instructor batch results --batch-id "batch_abc123" --output-file results.jsonl --model "openai/gpt-4.1-mini"
```

## Best Practices

1. **Batch Size**: Include at least 25,000 requests per job for optimal efficiency (especially for Google)
2. **Cost Optimization**: Use batch processing for non-urgent workloads to get 50% cost savings
3. **Error Handling**: Always check both successful and error results
4. **Timeout**: Batch jobs have execution limits (24 hours for Google, varies by provider)
5. **Storage**: For Google GenAI, ensure your GCS bucket is in the same region as your batch job

## Limitations

- **Google GenAI**: Requires Google Cloud Storage setup and proper IAM permissions
- **Processing Time**: Batch jobs are not real-time and can take significant time to complete
- **Queuing**: Jobs may be queued during high-demand periods
- **Regional Requirements**: Some providers require resources in specific regions

## Error Troubleshooting

### Google GenAI Common Issues
- **Missing GCS_BUCKET**: Set the `GCS_BUCKET` environment variable
- **Permission Denied**: Ensure service account has `roles/aiplatform.user` and `roles/storage.objectUser`
- **Region Mismatch**: Ensure GCS bucket and batch job are in the same region

### General Issues
- **Invalid Model Name**: Use format `provider/model-name` (e.g., `openai/gpt-4.1-mini`)
- **Authentication**: Ensure API keys are set correctly
- **Schema Validation**: Verify your Pydantic models match expected output format

For more details on Google Cloud batch prediction, see the [official documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini).
