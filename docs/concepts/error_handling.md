---
title: Error Handling in Instructor
description: Learn how to effectively handle errors and exceptions when using Instructor for structured outputs.
---

# Error Handling

Instructor provides a comprehensive exception hierarchy to help you handle errors gracefully and debug issues effectively. This guide covers the various exception types and best practices for error handling.

## Exception Hierarchy

All Instructor-specific exceptions inherit from `InstructorError`, making it easy to catch all Instructor-related errors:

```python
from instructor.exceptions import (
    InstructorError,
    IncompleteOutputException,
    InstructorRetryException,
    ValidationError,
    ProviderError,
    ConfigurationError,
    ModeError,
    ClientError
)
```

### Base Exception

- **`InstructorError`**: Base exception for all Instructor-specific errors. Catch this to handle any Instructor error.

### Specific Exception Types

#### `IncompleteOutputException`
Raised when the LLM output is incomplete due to reaching the maximum token limit.

```python
try:
    response = client.chat.completions.create(
        response_model=DetailedReport,
        messages=[{"role": "user", "content": "Write a very long report..."}],
        max_tokens=50  # Very low limit
    )
except IncompleteOutputException as e:
    print(f"Output was truncated: {e}")
    print(f"Last completion: {e.last_completion}")
```

#### `InstructorRetryException`
Raised when all retry attempts have been exhausted.

```python
try:
    response = client.chat.completions.create(
        response_model=UserDetail,
        messages=[{"role": "user", "content": "Extract user info..."}],
        max_retries=3
    )
except InstructorRetryException as e:
    print(f"Failed after {e.n_attempts} attempts")
    print(f"Last error: {e}")
    print(f"Last completion: {e.last_completion}")
    print(f"Total usage: {e.total_usage}")
```

#### `ValidationError`
Raised when response validation fails. This is different from Pydantic's ValidationError and provides additional context.

```python
try:
    response = client.chat.completions.create(
        response_model=StrictModel,
        messages=[{"role": "user", "content": "Extract data..."}]
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
```

#### `ProviderError`
Raised for provider-specific errors, includes the provider name for context.

```python
try:
    client = instructor.from_provider(invalid_client)
except ProviderError as e:
    print(f"Provider {e.provider} error: {e}")
```

#### `ConfigurationError`
Raised for configuration-related issues like invalid parameters or missing dependencies.

```python
try:
    client = instructor.from_provider("invalid/model")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

#### `ModeError`
Raised when an invalid mode is used for a specific provider.

```python
try:
    client = instructor.from_provider(
        "anthropic/claude-3-sonnet-20240229",
        mode=instructor.Mode.TOOLS,  # Wrong mode for Anthropic
    )
except ModeError as e:
    print(f"Invalid mode '{e.mode}' for provider '{e.provider}'")
    print(f"Valid modes: {', '.join(e.valid_modes)}")
```

#### `ClientError`
Raised for client initialization or usage errors.

```python
try:
    client = instructor.from_provider("not_a_client")
except ClientError as e:
    print(f"Client error: {e}")
```

## Best Practices

### 1. Catch Specific Exceptions When Possible

```python
from instructor.exceptions import (
    IncompleteOutputException,
    InstructorRetryException,
    ValidationError
)

try:
    response = client.chat.completions.create(...)
except IncompleteOutputException as e:
    # Handle truncated output - maybe increase max_tokens
    logger.warning(f"Output truncated, retrying with more tokens")
    response = client.chat.completions.create(..., max_tokens=2000)
except InstructorRetryException as e:
    # Handle retry exhaustion - maybe fallback logic
    logger.error(f"Failed after {e.n_attempts} attempts")
    return None
except ValidationError as e:
    # Handle validation errors - maybe relax constraints
    logger.error(f"Validation failed: {e}")
    raise
```

### 2. Use the Base Exception for General Error Handling

```python
from instructor.exceptions import InstructorError

try:
    response = client.chat.completions.create(...)
except InstructorError as e:
    # Catches any Instructor-specific error
    logger.error(f"Instructor error: {type(e).__name__}: {e}")
    raise
```

### 3. Handle Provider Setup Errors

```python
from instructor.exceptions import ConfigurationError, ClientError, ModeError

def create_client(provider: str, mode: str = None):
    try:
        client = instructor.from_provider(provider)
        return client
    except ConfigurationError as e:
        print(f"Configuration issue: {e}")
        # Maybe guide user to install missing package
    except ModeError as e:
        print(f"Invalid mode. Valid modes for {e.provider}: {e.valid_modes}")
        # Retry with a valid mode
    except ClientError as e:
        print(f"Client initialization failed: {e}")
        # Check client setup
```

### 4. Logging and Monitoring

```python
import logging
from instructor.exceptions import InstructorError

logger = logging.getLogger(__name__)

def extract_data(content: str):
    try:
        return client.chat.completions.create(
            response_model=DataModel,
            messages=[{"role": "user", "content": content}]
        )
    except InstructorError as e:
        logger.exception(
            "Failed to extract data",
            extra={
                "error_type": type(e).__name__,
                "provider": getattr(e, 'provider', None),
                "attempts": getattr(e, 'n_attempts', None),
            }
        )
        raise
```

### 5. Graceful Degradation

```python
from instructor.exceptions import ValidationError, InstructorRetryException

def extract_with_fallback(content: str):
    # Try with strict model first
    try:
        return client.chat.completions.create(
            response_model=StrictDataModel,
            messages=[{"role": "user", "content": content}]
        )
    except ValidationError:
        # Fall back to less strict model
        logger.warning("Strict validation failed, trying relaxed model")
        return client.chat.completions.create(
            response_model=RelaxedDataModel,
            messages=[{"role": "user", "content": content}]
        )
    except InstructorRetryException:
        # Final fallback
        logger.error("All retries exhausted, returning None")
        return None
```

## Integration with Hooks

Instructor's hooks system can be used to monitor and handle errors programmatically:

```python
from instructor import Instructor
from instructor.exceptions import ValidationError

def on_parse_error(error: Exception):
    if isinstance(error, ValidationError):
        # Log validation errors to monitoring service
        monitoring.log_validation_error(str(error))

client = Instructor(...)
client.hooks.on("parse:error", on_parse_error)
```

## Common Error Scenarios

### Missing Dependencies

```python
try:
    client = instructor.from_provider("anthropic/claude-3")
except ConfigurationError as e:
    if "package is required" in str(e):
        print("Please install the anthropic package: pip install anthropic")
```

### Invalid Provider Format

```python
try:
    client = instructor.from_provider("invalid-format")
except ConfigurationError as e:
    print(e)  # Model string must be in format "provider/model-name"
```

### Unsupported Mode

```python
try:
    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.ANTHROPIC_TOOLS,  # Wrong mode
    )
except ModeError as e:
    print(f"Use one of these modes instead: {e.valid_modes}")
```

## See Also

- [Retrying](./retrying.md) - Learn about retry strategies
- [Validation](./validation.md) - Understanding validation in Instructor
- [Hooks](./hooks.md) - Using hooks for error monitoring