---
title: "Structured outputs with llama-cpp-python, a complete guide w/ instructor"
description: "Complete guide to using Instructor with llama-cpp-python for local LLM deployment. Learn about performance considerations, limitations, and best practices for structured outputs."
---

# Structured outputs with llama-cpp-python, a complete guide w/ instructor

llama-cpp-python provides Python bindings for llama.cpp, enabling local deployment of LLMs. This guide shows you how to use Instructor with llama-cpp-python for type-safe, validated responses while being aware of important performance considerations and limitations.

## Important Limitations

Before getting started, be aware of these critical limitations:

### Performance Considerations
- **CPU-Only Execution**: Currently runs on CPU only, which significantly impacts performance
- **Long Inference Times**: Expect 30-60+ seconds for simple extractions on CPU
- **Context Window Management**:
  - Default context size is 2048 tokens (configurable)
  - Larger contexts (>4096) may require more memory
  - Adjust n_ctx based on your needs and available memory
- **Memory Usage**: Requires ~4GB of RAM for model loading

### Streaming Support
- **Basic Streaming**: ✓ Supported and verified working
- **Structured Output Streaming**: ✓ Supported with limitations
  - Chunks are delivered in larger intervals compared to cloud providers
  - Response time may be slower due to CPU-only processing
  - Partial objects stream correctly but with higher latency
- **Async Support**: ❌ Not supported (AsyncLlama is not available)

## Quick Start

Install Instructor with llama-cpp-python support:

```bash
pip install "instructor[llama-cpp-python]"
```

## Simple User Example (Sync)

```python
from llama_cpp import Llama
from instructor import patch
from pydantic import BaseModel

# Initialize the model with appropriate settings
llm = Llama(
    model_path="path/to/your/gguf/model",
    n_ctx=2048,  # Adjust based on your needs and memory constraints
    n_batch=32  # Adjust for performance vs memory trade-off
)

# Enable instructor patches
client = patch(llm)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.chat.create(
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_model=User,
    max_tokens=100,
    temperature=0.1
)

print(user)  # User(name='Jason', age=25)
```

## Nested Example

```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]

# Create structured output with nested objects
user = client.chat.create(
    messages=[{
        "role": "user",
        "content": """
            Extract: Jason is 25 years old.
            He lives at 123 Main St, New York, USA
            and has a summer house at 456 Beach Rd, Miami, USA
        """
    }],
    response_model=User,
    max_tokens=200,
    temperature=0.1
)

print(user)  # User with nested Address objects
```

## Partial Streaming Example

```python
class User(BaseModel):
    name: str
    age: int
    bio: str

# Stream partial objects as they're generated
for partial_user in client.chat.create(
    messages=[{"role": "user", "content": "Create a user profile for Jason, age 25"}],
    response_model=User,
    max_tokens=100,
    temperature=0.1,
    stream=True
):
    print(f"Current state: {partial_user}")
    # Fields will populate gradually as they're generated
```

## Iterable Example

```python
from typing import List

class User(BaseModel):
    name: str
    age: int

# Extract multiple users from text
users = client.chat.create(
    messages=[{
        "role": "user",
        "content": """
            Extract users:
            1. Jason is 25 years old
            2. Sarah is 30 years old
            3. Mike is 28 years old
        """
    }],
    response_model=User,
    max_tokens=100,
    temperature=0.1
)

for user in users:
    print(user)  # Prints each user as it's extracted
```

## Instructor Hooks

Instructor provides several hooks to customize behavior:

### Validation Hook

```python
from instructor import patch

def validation_hook(value, retry_count, exception):
    print(f"Validation failed {retry_count} times: {exception}")
    return retry_count < 3  # Retry up to 3 times

patch(client, validation_hook=validation_hook)
```

### Mode Hooks

```python
from instructor import Mode

# Use different modes for different scenarios
client = patch(client, mode=Mode.JSON)  # JSON mode
client = patch(client, mode=Mode.TOOLS)  # Tools mode
client = patch(client, mode=Mode.MD_JSON)  # Markdown JSON mode
```

### Custom Retrying

```python
from instructor import RetryConfig

client = patch(
    client,
    retry_config=RetryConfig(
        max_retries=3,
        on_retry=lambda *args: print("Retrying..."),
    )
)
```

## Model Configuration and Performance Considerations

### Hardware Requirements and Limitations
- **CPU-Only Operation**: Currently, the implementation runs on CPU only
- **Memory Usage**: Requires approximately 4GB RAM for model loading
- **Processing Speed**: Expect significant processing times (30-60+ seconds) for simple extractions

### Key Configuration Options
- `n_ctx`: Context window size (default: 2048, limited compared to training context of 4096)
- `n_batch`: Batch size for prompt processing (adjust for memory/performance trade-off)
- `n_threads`: Number of CPU threads to use (optimize based on your hardware)

## Best Practices

1. **Resource Management**
   - Monitor CPU usage and memory consumption
   - Keep prompts concise due to context window limitations
   - Implement appropriate timeouts for long-running operations
   - Consider request queuing for multiple users

2. **Model Selection**
   - Use quantized models to reduce memory usage
   - Balance model size vs performance needs
   - Consider smaller models for faster inference
   - Test with your specific use case

3. **Performance Optimization**
   - Batch similar requests when possible
   - Implement caching strategies
   - Use appropriate timeout values
   - Monitor and log performance metrics

## Common Use Cases

- Local Development
- Privacy-Sensitive Applications
- Edge Computing
- Offline Processing
- Resource-Constrained Environments

## Related Resources

- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with the latest llama-cpp-python releases. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.
