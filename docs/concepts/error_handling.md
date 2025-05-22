# Enhanced Error Handling

Instructor provides comprehensive error handling with structured error messages, actionable guidance, and debugging context to help you quickly identify and resolve issues.

## Error Hierarchy

All Instructor exceptions inherit from `InstructorError`, making it easy to catch any Instructor-specific error:

```python
from instructor.exceptions import InstructorError

try:
    # Your Instructor code
    response = client.chat.completions.create(...)
except InstructorError as e:
    print(f"Error: {e}")
    print(f"Guidance: {e.guidance}")
    print(f"Details: {e.details}")
```

## Exception Types

### ValidationError

Raised when Pydantic validation fails. Includes specific guidance based on the validation errors:

```python
from instructor.exceptions import ValidationError

try:
    response = client.chat.completions.create(
        model="gpt-4",
        response_model=UserDetail,
        messages=[{"role": "user", "content": "Invalid data"}],
    )
except ValidationError as e:
    print(e)
    # Output:
    # ValidationError: Validation failed for UserDetail
    # Details: {
    #   "validation_errors": [
    #     {"loc": ["name"], "type": "missing", "msg": "Field required"}
    #   ]
    # }
    # 
    # Guidance: Validation issues found:
    # - Field 'name' is required. Ensure your prompt asks for this field explicitly.
```

### ConfigurationError

Raised for configuration issues with helpful guidance:

```python
from instructor.exceptions import ConfigurationError

try:
    client = instructor.from_openai(OpenAI(api_key=None))
except ConfigurationError as e:
    print(e.guidance)
    # Output: Set your API key via environment variable or pass it to the client constructor.
```

### ParsingError

Raised when response parsing fails, with guidance on fixing the issue:

```python
from instructor.exceptions import ParsingError

# If parsing fails
except ParsingError as e:
    print(e.guidance)
    # Output:
    # - Response doesn't appear to be JSON. Ensure you're using JSON mode.
    # - Try using a more specific prompt that requests JSON output
    # - Consider using Mode.TOOLS instead of Mode.JSON for better reliability
```

### TokenLimitError

Specific error for token limit issues:

```python
from instructor.exceptions import TokenLimitError

except TokenLimitError as e:
    print(e)
    # Output:
    # TokenLimitError: Maximum token limit exceeded
    # Details: {"token_count": 5000, "token_limit": 4096}
    # 
    # Guidance: Token limit exceeded. Try:
    # - Current: 5000 tokens, Limit: 4096 tokens
    # - Reducing the prompt length
    # - Using a smaller response model
    # - Breaking the task into smaller chunks
```

### Provider-Specific Errors

Each provider has its own error class for provider-specific issues:

```python
from instructor.exceptions import OpenAIError, AnthropicError

try:
    # OpenAI specific operation
    pass
except OpenAIError as e:
    print(f"OpenAI error: {e}")
    print(f"Provider: {e.provider}")  # "openai"
```

## Error Context

Errors include request and response context for debugging:

```python
try:
    response = client.chat.completions.create(...)
except InstructorError as e:
    # Access debugging context
    print(f"Request context: {e.request_context}")
    print(f"Response context: {e.response_context}")
    print(f"Timestamp: {e.timestamp}")
    
    # Convert to dict for logging
    error_dict = e.to_dict()
    logger.error("Instructor error", extra=error_dict)
```

## Structured Logging

All errors can be converted to dictionaries for structured logging:

```python
import logging
import json

logger = logging.getLogger(__name__)

try:
    response = client.chat.completions.create(...)
except InstructorError as e:
    # Log structured error
    logger.error(
        "Instructor operation failed",
        extra={
            "error_data": e.to_dict(),
            "operation": "chat_completion",
            "model": "gpt-4",
        }
    )
```

## Creating Custom Errors

You can create custom errors with the factory function:

```python
from instructor.exceptions import create_error_from_response

# From a provider response
error = create_error_from_response(
    provider="openai",
    response=api_response,
    request_context={"model": "gpt-4", "temperature": 0.7}
)

# The factory automatically:
# - Detects error type (token limit, rate limit, etc.)
# - Adds appropriate guidance
# - Extracts error codes and details
```

## Best Practices

### 1. Catch Specific Exceptions

```python
from instructor.exceptions import ValidationError, TokenLimitError, RetryError

try:
    response = client.chat.completions.create(...)
except ValidationError as e:
    # Handle validation errors
    # Maybe adjust the prompt or model
    pass
except TokenLimitError as e:
    # Handle token limits
    # Maybe break into smaller chunks
    pass
except RetryError as e:
    # Handle retry exhaustion
    # Maybe increase retries or use different model
    pass
```

### 2. Use Error Guidance

The guidance provided in exceptions is designed to be actionable:

```python
try:
    response = client.chat.completions.create(...)
except InstructorError as e:
    if e.guidance:
        print(f"💡 Suggestion: {e.guidance}")
        # Implement the suggested fix
```

### 3. Log Error Context

Always log the full error context for debugging:

```python
import structlog

logger = structlog.get_logger()

try:
    response = client.chat.completions.create(...)
except InstructorError as e:
    logger.error(
        "instructor_error",
        **e.to_dict(),
        user_id=current_user.id,
        feature="data_extraction"
    )
```

### 4. Handle Legacy Exceptions

For backward compatibility, legacy exceptions still work:

```python
from instructor.exceptions import IncompleteOutputException

try:
    response = client.chat.completions.create(...)
except IncompleteOutputException:
    # Legacy exception still caught
    # But you get enhanced error features
    pass
```

## Error Recovery Strategies

Based on the error type, here are recommended recovery strategies:

| Error Type | Recovery Strategy |
|------------|------------------|
| ValidationError | Adjust prompt to be more specific about required fields |
| TokenLimitError | Reduce prompt/response size or switch to a model with higher limits |
| RetryError | Increase max_retries or improve prompt clarity |
| ParsingError | Switch from JSON mode to TOOLS mode for better reliability |
| ConfigurationError | Fix configuration based on the guidance provided |

## Debugging Tips

1. **Enable Debug Logging**: Set up logging to capture full error details
2. **Check Error Codes**: Use `error.error_code` for programmatic error handling
3. **Review Request Context**: Check what was sent to the API
4. **Follow Guidance**: The guidance is generated based on the specific error
5. **Use Timestamps**: Error timestamps help correlate with API logs

## Related Resources

- [Retrying Failed Requests](./retrying.md)
- [Validation](./validation.md)
- [Logging](./logging.md)