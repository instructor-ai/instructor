---
title: Error Handling in Instructor
description: Learn how to effectively handle errors and exceptions when using Instructor for structured outputs.
---

## See Also

- [Validation](./validation.md) - Core validation concepts and error handling
- [Retrying](./retrying.md) - Automatic retry mechanisms with Tenacity
- [Reask Validation](./reask_validation.md) - Automatic retry with validation feedback
- [Hooks](./hooks.md) - Monitor errors and retries with hooks
- [Debugging](../debugging.md) - Practical debugging techniques

# Error Handling

Instructor provides a comprehensive exception hierarchy to help you handle errors gracefully and debug issues effectively. This guide covers the various exception types and best practices for error handling.

## Exception Hierarchy

All Instructor-specific exceptions inherit from `InstructorError`, making it easy to catch all Instructor-related errors.

### Base Exception

- **`InstructorError`**: Base exception for all Instructor-specific errors. Catch this to handle any Instructor error.

### Specific Exception Types

#### `IncompleteOutputException`
Raised when the LLM output is incomplete due to reaching the maximum token limit.

```python
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import (
    IncompleteOutputException,
    InstructorRetryException,
)


class DetailedReport(BaseModel):
    content: str


client = instructor.from_provider(
    "openai/gpt-4.1-mini",
    mode=instructor.Mode.JSON,
)

try:
    response = client.create(
        response_model=DetailedReport,
        messages=[{"role": "user", "content": "Write a very long report..."}],
        max_tokens=50,  # Very low limit
        max_retries=0,
    )
except (IncompleteOutputException, InstructorRetryException) as e:
    print(f"Output was truncated: {e}")
    """
    Output was truncated: <failed_attempts>

    <generation number="1">
    <exception>
        The output is incomplete due to a max_tokens length limit.
    </exception>
    <completion>
        ChatCompletion(id='chatcmpl-D08pGj8uPNcBmPrmA77fqYN27lCON', choices=[Choice(finish_reason='length', index=0, logprobs=None, message=ChatCompletionMessage(content='{\n  "content": "This report aims to provide an in-depth exploration of various significant topics that impact our society and technological progress. It encompasses a comprehensive analysis of economic trends, technological advancements, environmental challenges, and social dynamics, all critical in shaping', refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None))], created=1768926298, model='gpt-4.1-mini-2025-04-14', object='chat.completion', service_tier='default', system_fingerprint='fp_376a7ccef1', usage=CompletionUsage(completion_tokens=50, prompt_tokens=120, total_tokens=170, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=None, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=None), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    </completion>
    </generation>

    </failed_attempts>

    <last_exception>
        The output is incomplete due to a max_tokens length limit.
    </last_exception>
    """
    print(f"Last completion: {e.last_completion}")
    """
    Last completion: ChatCompletion(id='chatcmpl-D08pGj8uPNcBmPrmA77fqYN27lCON', choices=[Choice(finish_reason='length', index=0, logprobs=None, message=ChatCompletionMessage(content='{\n  "content": "This report aims to provide an in-depth exploration of various significant topics that impact our society and technological progress. It encompasses a comprehensive analysis of economic trends, technological advancements, environmental challenges, and social dynamics, all critical in shaping', refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None))], created=1768926298, model='gpt-4.1-mini-2025-04-14', object='chat.completion', service_tier='default', system_fingerprint='fp_376a7ccef1', usage=CompletionUsage(completion_tokens=50, prompt_tokens=120, total_tokens=170, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=None, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=None), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    """
```

#### `InstructorRetryException`
Raised when all retry attempts have been exhausted.

```python
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import InstructorRetryException


class UserDetail(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "openai/gpt-4.1-mini",
    mode=instructor.Mode.JSON,
)

try:
    response = client.create(
        response_model=UserDetail,
        messages=[{"role": "user", "content": "Extract user info..."}],
        max_retries=3,
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
import instructor
from pydantic import BaseModel, field_validator
from instructor.core.exceptions import ValidationError


class StrictModel(BaseModel):
    value: int

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Value must be positive")
        return v


client = instructor.from_provider("openai/gpt-4.1-mini")

try:
    response = client.create(
        response_model=StrictModel,
        messages=[{"role": "user", "content": "Extract data..."}],
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
```

#### `ProviderError`
Raised for provider-specific errors, includes the provider name for context.

```python
import instructor
from instructor.core.exceptions import ProviderError, ConfigurationError

try:
    client = instructor.from_provider("invalid/provider")
except (ProviderError, ConfigurationError) as e:
    provider = getattr(e, "provider", "unknown")
    print(f"Provider {provider} error: {e}")
    """
    Provider unknown error: Unsupported provider: invalid. Supported providers are: ['openai', 'azure_openai', 'databricks', 'anthropic', 'google', 'generative-ai', 'vertexai', 'mistral', 'cohere', 'perplexity', 'groq', 'writer', 'bedrock', 'cerebras', 'deepseek', 'fireworks', 'ollama', 'openrouter', 'xai', 'litellm']
    """
```

#### `ConfigurationError`
Raised for configuration-related issues like invalid parameters or missing dependencies.

```python
import instructor
from instructor.core.exceptions import ConfigurationError

try:
    client = instructor.from_provider("invalid/model")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    """
    Configuration error: Unsupported provider: invalid. Supported providers are: ['openai', 'azure_openai', 'databricks', 'anthropic', 'google', 'generative-ai', 'vertexai', 'mistral', 'cohere', 'perplexity', 'groq', 'writer', 'bedrock', 'cerebras', 'deepseek', 'fireworks', 'ollama', 'openrouter', 'xai', 'litellm']
    """
```

#### `ModeError`
Raised when an invalid mode is used for a specific provider.

```python
import instructor
from instructor.core.exceptions import ModeError

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
import instructor
from instructor.core.exceptions import ClientError, ConfigurationError

try:
    client = instructor.from_provider("not_a_client")
except (ClientError, ConfigurationError) as e:
    print(f"Client error: {e}")
    """
    Client error: Model string must be in format "provider/model-name" (e.g. "openai/gpt-4" or "anthropic/claude-3-sonnet")
    """
```

#### `ResponseParsingError`
Raised when unable to parse the LLM response into the expected format. Inherits from both `ValueError` and `InstructorError` for backwards compatibility.

```python
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import ResponseParsingError


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4.1-mini")

try:
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Jane is 28"}],
    )
except ResponseParsingError as e:
    print(f"Failed to parse response in {e.mode} mode")
    print(f"Raw response: {e.raw_response}")
    # Access diagnostic information
```

**Key Attributes:**
- `mode`: The mode being used when parsing failed
- `raw_response`: The raw LLM response for debugging

**Backwards Compatible:** Can still be caught as `ValueError` for existing code.

#### `MultimodalError`
Raised when processing multimodal content (images, audio, PDFs) fails. Inherits from both `ValueError` and `InstructorError` for backwards compatibility.

```python
import instructor
from instructor import Image
from pydantic import BaseModel
from instructor.core.exceptions import MultimodalError


class Analysis(BaseModel):
    summary: str


client = instructor.from_provider("openai/gpt-4.1-mini")

try:
    response = client.create(
        response_model=Analysis,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this"},
                    Image.from_path("tests/assets/image.jpg"),
                ],
            }
        ],
    )
except MultimodalError as e:
    print(f"Multimodal error with {e.content_type}: {e}")
    if e.file_path:
        print(f"File: {e.file_path}")
```

**Key Attributes:**
- `content_type`: The type of content (e.g., 'image', 'audio', 'pdf')
- `file_path`: The file path if applicable

**Backwards Compatible:** Can still be caught as `ValueError` for existing code.

#### `AsyncValidationError`
Raised during async validation operations. Inherits from both `ValueError` and `InstructorError`.

```python
import asyncio
import instructor
from pydantic import BaseModel
from instructor.validation import async_field_validator
from instructor.core.exceptions import AsyncValidationError


class Model(BaseModel):
    urls: list[str]

    @async_field_validator('urls')
    async def validate_urls(cls, v):
        # Async validation logic
        ...


async def main() -> None:
    client = instructor.from_provider("openai/gpt-4.1-mini", async_client=True)
    try:
        await client.create(
            response_model=Model,
            messages=[{"role": "user", "content": "Validate these URLs"}],
        )
    except AsyncValidationError as e:
        print(f"Async validation failed: {e.errors}")


asyncio.run(main())
```

## Best Practices

### 1. Catch Specific Exceptions When Possible

```python
import logging
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import (
    IncompleteOutputException,
    InstructorRetryException,
    ValidationError,
)


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4.1-mini")
logger = logging.getLogger(__name__)

try:
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Sam is 34"}],
    )
except IncompleteOutputException as e:
    # Handle truncated output - maybe increase max_tokens
    logger.warning(f"Output truncated, retrying with more tokens")
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Sam is 34"}],
        max_tokens=2000,
    )
except InstructorRetryException as e:
    # Handle retry exhaustion - maybe fallback logic
    logger.error(f"Failed after {e.n_attempts} attempts")
    response = None
except ValidationError as e:
    # Handle validation errors - maybe relax constraints
    logger.error(f"Validation failed: {e}")
    raise
```

### 2. Use the Base Exception for General Error Handling

```python
import logging
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import InstructorError


class DataModel(BaseModel):
    value: str


client = instructor.from_provider("openai/gpt-4.1-mini")
logger = logging.getLogger(__name__)

try:
    response = client.create(
        response_model=DataModel,
        messages=[{"role": "user", "content": "Extract data"}],
    )
except InstructorError as e:
    # Catches any Instructor-specific error
    logger.error(f"Instructor error: {type(e).__name__}: {e}")
    raise
```

### 3. Handle Provider Setup Errors

```python
import instructor
from instructor.core.exceptions import ConfigurationError, ClientError, ModeError


def create_client(provider: str, _mode: str | None = None):
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
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import InstructorError

logger = logging.getLogger(__name__)


class DataModel(BaseModel):
    value: str


client = instructor.from_provider("openai/gpt-4.1-mini")


def extract_data(content: str):
    try:
        return client.create(
            response_model=DataModel,
            messages=[{"role": "user", "content": content}],
        )
    except InstructorError as e:
        logger.exception(
            "Failed to extract data",
            extra={
                "error_type": type(e).__name__,
                "provider": getattr(e, 'provider', None),
                "attempts": getattr(e, 'n_attempts', None),
            },
        )
        raise
```

### 5. Graceful Degradation

```python
import logging
import instructor
from pydantic import BaseModel, field_validator
from instructor.core.exceptions import ValidationError, InstructorRetryException

logger = logging.getLogger(__name__)


class StrictDataModel(BaseModel):
    value: int

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Value must be positive")
        return v


class RelaxedDataModel(BaseModel):
    value: str


client = instructor.from_provider("openai/gpt-4.1-mini")


def extract_with_fallback(content: str):
    # Try with strict model first
    try:
        return client.create(
            response_model=StrictDataModel,
            messages=[{"role": "user", "content": content}],
        )
    except ValidationError:
        # Fall back to less strict model
        logger.warning("Strict validation failed, trying relaxed model")
        return client.create(
            response_model=RelaxedDataModel,
            messages=[{"role": "user", "content": content}],
        )
    except InstructorRetryException:
        # Final fallback
        logger.error("All retries exhausted, returning None")
        return None
```

## Backwards Compatibility

All new exception types maintain backwards compatibility by inheriting from both `ValueError` and `InstructorError`:

```python
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import ResponseParsingError, MultimodalError


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4.1-mini")

# Old code still works - catching ValueError
try:
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Kai is 41"}],
    )
except ValueError as e:
    # Will catch ResponseParsingError and MultimodalError
    print(f"Error: {e}")

# New code can be more specific
try:
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Kai is 41"}],
    )
except ResponseParsingError as e:
    # Access additional context
    print(f"Mode: {e.mode}")
    print(f"Raw response: {e.raw_response}")
except MultimodalError as e:
    print(f"Content type: {e.content_type}")
    print(f"File path: {e.file_path}")
```

## Diagnostic Context

New exceptions include rich diagnostic context for monitoring and debugging:

### Response Parsing Errors

```python
import logging
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import ResponseParsingError


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4.1-mini")
logger = logging.getLogger(__name__)

try:
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Alex is 22"}],
    )
except ResponseParsingError as e:
    # Log full context for monitoring
    logger.error(
        "Response parsing failed",
        extra={
            "mode": e.mode,
            "raw_response": e.raw_response,
            "error_message": str(e),
        },
    )
```

### Multimodal Errors

```python
import logging
from instructor import Image
from instructor.core.exceptions import MultimodalError

logger = logging.getLogger(__name__)

try:
    img = Image.from_path("tests/assets/image.jpg")
except MultimodalError as e:
    # Log with file context
    logger.error(
        "Multimodal content error",
        extra={
            "content_type": e.content_type,
            "file_path": e.file_path,
            "error_message": str(e),
        },
    )
```

### Retry Exceptions

```python
import logging
import instructor
from pydantic import BaseModel
from instructor.core.exceptions import InstructorRetryException


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4.1-mini")
logger = logging.getLogger(__name__)

try:
    response = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Riley is 27"}],
    )
except InstructorRetryException as e:
    # Access all failed attempts for analysis
    for attempt in e.failed_attempts:
        logger.warning(
            f"Attempt {attempt.attempt_number} failed",
            extra={
                "exception": str(attempt.exception),
                "partial_completion": attempt.completion,
            },
        )

    # Log final failure with full context
    logger.error(
        "All retries exhausted",
        extra={
            "n_attempts": e.n_attempts,
            "total_usage": e.total_usage,
            "model": e.create_kwargs.get("model"),
            "last_completion": e.last_completion,
        },
    )
```

## Integration with Hooks

Instructor's hooks system can be used to monitor and handle errors programmatically:

```python
import instructor
from instructor.core.exceptions import ValidationError


class Monitoring:
    def log_validation_error(self, message: str) -> None: ...


monitoring = Monitoring()


def on_parse_error(error: Exception):
    if isinstance(error, ValidationError):
        # Log validation errors to monitoring service
        monitoring.log_validation_error(str(error))


client = instructor.from_provider("openai/gpt-4.1-mini")
client.hooks.on("parse:error", on_parse_error)
```

## Common Error Scenarios

### Missing Dependencies

```python
import instructor
from instructor.core.exceptions import ConfigurationError

try:
    client = instructor.from_provider("anthropic/claude-3")
except ConfigurationError as e:
    if "package is required" in str(e):
        print("Please install the anthropic package: pip install anthropic")
```

### Invalid Provider Format

```python
import instructor
from instructor.core.exceptions import ConfigurationError

try:
    client = instructor.from_provider("invalid-format")
except ConfigurationError as e:
    print(e)  # Model string must be in format "provider/model-name"
    """
    Model string must be in format "provider/model-name" (e.g. "openai/gpt-4" or "anthropic/claude-3-sonnet")
    """
```

### Unsupported Mode

```python
import instructor
from instructor.core.exceptions import ModeError

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
