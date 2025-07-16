---
title: Understanding Hooks in the Instructor Library
description: Learn how to use hooks for event handling in the Instructor library to enhance logging, error handling, and custom behaviors.
---

# Hooks

Hooks provide a powerful mechanism for intercepting and handling events during the completion and parsing process in the Instructor library. They allow you to add custom behavior, logging, or error handling at various stages of the API interaction.

## Overview

The Hooks system in Instructor is based on the `Hooks` class, which manages event registration and emission. It supports several predefined events that correspond to different stages of the completion and parsing process.

## Supported Hook Events

### `completion:kwargs`

This hook is emitted when completion arguments are provided. It receives all arguments passed to the completion function. These will contain the `model`, `messages`, `tools`, AFTER any `response_model` or `validation_context` parameters have been converted to their respective values.

```python
def handler(*args, **kwargs) -> None: ...
```

### `completion:response`

This hook is emitted when a completion response is received. It receives the raw response object from the completion API.

```python
def handler(response) -> None: ...
```

### `completion:error`

This hook is emitted when an error occurs during completion before any retries are attempted and the response is parsed as a pydantic model.

```python
def handler(error) -> None: ...
```

### `parse:error`

This hook is emitted when an error occurs during parsing of the response as a pydantic model. This can happen if the response is not valid or if the pydantic model is not compatible with the response.

```python
def handler(error) -> None: ...
```

### `completion:last_attempt`

This hook is emitted when the last retry attempt is made.

```python
def handler(error) -> None: ...
```

## Implementation Details

The Hooks system is implemented in the `instructor/hooks.py` file. The `Hooks` class handles the registration and emission of hook events. You can refer to this file to see how hooks work under the hood.
The retry logic that uses Hooks is implemented in the `instructor/retry.py` file. This shows how Hooks are used when trying again after errors during completions.

### Hook Types

The Hooks system uses typed Protocol classes to provide better type safety for handler functions:

```python
from typing import Any, Protocol


# Handler protocol types for type safety
class CompletionKwargsHandler(Protocol):
    """Protocol for completion kwargs handlers."""

    def __call__(self, *args: Any, **kwargs: Any) -> None: ...


class CompletionResponseHandler(Protocol):
    """Protocol for completion response handlers."""

    def __call__(self, response: Any) -> None: ...


class CompletionErrorHandler(Protocol):
    """Protocol for completion error and last attempt handlers."""

    def __call__(self, error: Exception) -> None: ...


class ParseErrorHandler(Protocol):
    """Protocol for parse error handlers."""

    def __call__(self, error: Exception) -> None: ...
```

These Protocol types help ensure that your handler functions have the correct signature for each type of hook.

### Hook Names

Hook names can be specified either as enum values (`HookName.COMPLETION_KWARGS`) or as strings (`"completion:kwargs"`):

```python
from instructor.hooks import HookName

# Using enum
client.on(HookName.COMPLETION_KWARGS, handler)

# Using string
client.on("completion:kwargs", handler)
```

### Registering Hooks

You can register hooks using the `on` method of the Instructor client or a `Hooks` instance. Here's an example:

```python
import instructor
import pprint

client = instructor.from_provider("openai/gpt-4.1-mini")


def log_completion_kwargs(*args, **kwargs):
    pprint.pprint({"args": args, "kwargs": kwargs})


client.on("completion:kwargs", log_completion_kwargs)

resp = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello, world!"}],
    response_model=str,
)
print(resp)
#> Hello, user! How can I assist you today?
```

### Emitting Events

Events are automatically emitted by the Instructor library at appropriate times. You don't need to manually emit events in most cases. Internally, all emit methods use a common `emit` method that handles error trapping and provides consistent behavior.

### Removing Hooks

You can remove a specific hook using the `off` method:

```python
import instructor
import pprint

client = instructor.from_provider("openai/gpt-4.1-mini")


def log_completion_kwargs(*args, **kwargs):
    pprint.pprint({"args": args, "kwargs": kwargs})


# Register the hook
client.on("completion:kwargs", log_completion_kwargs)

# Then later, remove it when no longer needed
client.off("completion:kwargs", log_completion_kwargs)
```

### Clearing Hooks

To remove all hooks for a specific event or all events:

```python
import instructor

client = instructor.from_provider("openai/gpt-4.1-mini")

# Define a simple handler
def log_completion_kwargs(*args, **kwargs):
    print("Logging completion kwargs...")

# Register the hook
client.on("completion:kwargs", log_completion_kwargs)

# Make a request that triggers the hook
resp = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello, world!"}],
    response_model=str,
)

# Clear hooks for a specific event
client.clear("completion:kwargs")

# Register another handler for a different event
def log_response(response):
    print("Logging response...")

client.on("completion:response", log_response)

# Clear all hooks
client.clear()
```

## Example: Logging and Debugging

Here's a comprehensive example demonstrating how to use hooks for logging and debugging:

```python
import instructor
import pydantic


def log_completion_kwargs(*args, **kwargs) -> None:
    print("## Completion kwargs:")
    print(kwargs)
    # Example output:
    # {
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": "Extract the user name and age from the following text: 'John is 20 years old'",
    #         }
    #     ],
    #     "model": "gpt-4.1-mini",
    #     "tools": [
    #         {
    #             "type": "function",
    #             "function": {
    #                 "name": "User",
    #                 "description": "Correctly extracted `User` with all the required parameters with correct types",
    #                 "parameters": {
    #                     "properties": {
    #                         "name": {"title": "Name", "type": "string"},
    #                         "age": {"title": "Age", "type": "integer"},
    #                     },
    #                     "required": ["age", "name"],
    #                     "type": "object",
    #                 },
    #             },
    #         }
    #     ],
    #     "tool_choice": {"type": "function", "function": {"name": "User"}},
    # }


def log_completion_response(response) -> None:
    print("## Completion response:")
    # Example output:
    # {
    #     'id': 'chatcmpl-AWl4Mj5Jrv7m7JkOTIiHXSldQIOFm',
    #     'choices': [
    #         {
    #             'finish_reason': 'stop',
    #             'index': 0,
    #             'logprobs': None,
    #             'message': {
    #                 'content': None,
    #                 'refusal': None,
    #                 'role': 'assistant',
    #                 'audio': None,
    #                 'function_call': None,
    #                 'tool_calls': [
    #                     {
    #                         'id': 'call_6oQ9WXxeSiVEV71B9IYtsbIE',
    #                         'function': {
    #                             'arguments': '{"name":"John","age":-1}',
    #                             'name': 'User',
    #                         },
    #                         'type': 'function',
    #                     }
    #                 ],
    #             },
    #         }
    #     ],
    #     'created': 1732370794,
    #     'model': 'gpt-4.1-mini-2024-07-18',
    #     'object': 'chat.completion',
    #     'service_tier': None,
    #     'system_fingerprint': 'fp_0705bf87c0',
    #     'usage': {
    #         'completion_tokens': 10,
    #         'prompt_tokens': 87,
    #         'total_tokens': 97,
    #         'completion_tokens_details': {
    #             'audio_tokens': 0,
    #             'reasoning_tokens': 0,
    #             'accepted_prediction_tokens': 0,
    #             'rejected_prediction_tokens': 0,
    #         },
    #         'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0},
    #     },
    # }
    print(response.model_dump())


def handle_completion_error(error: Exception) -> None:
    print(f"## Completion error: {error}")
    print(f"Type: {type(error).__name__}")
    print(f"Message: {str(error)}")

    # Handle specific Instructor exceptions
    from instructor.exceptions import (
        IncompleteOutputException,
        ValidationError,
        ProviderError
    )

    if isinstance(error, IncompleteOutputException):
        print(f"Output was truncated. Last completion: {error.last_completion}")
    elif isinstance(error, ValidationError):
        print("Validation failed - check your model schema")
    elif isinstance(error, ProviderError):
        print(f"Provider {error.provider} had an issue")


def log_parse_error(error: Exception) -> None:
    print(f"## Parse error: {error}")
    print(f"Type: {type(error).__name__}")
    print(f"Message: {str(error)}")

    # You can also check for Pydantic validation errors
    from pydantic import ValidationError as PydanticValidationError
    if isinstance(error, PydanticValidationError):
        print("Pydantic validation errors:")
        for err in error.errors():
            print(f"  - {err['loc']}: {err['msg']}")


# Handler for a custom logger that records how many errors have occurred
class ErrorCounter:
    def __init__(self) -> None:
        self.error_count = 0

    def count_error(self, error: Exception) -> None:
        self.error_count += 1
        print(f"Error count: {self.error_count}")


client = instructor.from_provider("openai/gpt-4.1-mini")

# Register the hooks
client.on("completion:kwargs", log_completion_kwargs)
client.on("completion:response", log_completion_response)
client.on("completion:error", handle_completion_error)
client.on("parse:error", log_parse_error)

# Example with error counter
error_counter = ErrorCounter()
client.on("completion:error", error_counter.count_error)
client.on("parse:error", error_counter.count_error)

# Define a model for extraction
class User(pydantic.BaseModel):
    name: str
    age: int

# Try extraction with a potentially problematic input
try:
    resp = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Extract the user name and age: 'John is twenty years old'",
            }
        ],
        response_model=User,
    )
    print(f"Extracted: {resp}")
except Exception as e:
    print(f"Main exception caught: {e}")

# Check the error count
print(f"Total errors recorded: {error_counter.error_count}")
```

## Advanced: Creating Custom Hooks

While the Instructor library provides several built-in hooks, you might need to create custom hooks for specific use cases. You can do this by extending the `HookName` enum and adding handlers for your custom events:

```python
from typing import Protocol, Any
from enum import Enum
from instructor.hooks import Hooks, HookName


# Extend the HookName enum
class CustomHookName(str, Enum):
    CUSTOM_EVENT = "custom:event"

    # Make it compatible with the base HookName enum
    COMPLETION_KWARGS = HookName.COMPLETION_KWARGS.value
    COMPLETION_RESPONSE = HookName.COMPLETION_RESPONSE.value
    COMPLETION_ERROR = HookName.COMPLETION_ERROR.value
    PARSE_ERROR = HookName.PARSE_ERROR.value
    COMPLETION_LAST_ATTEMPT = HookName.COMPLETION_LAST_ATTEMPT.value


# Create a hooks instance
hooks = Hooks()


# Define a handler
def custom_handler(data):
    print(f"Custom event: {data}")


# Register the handler
hooks.on(CustomHookName.CUSTOM_EVENT, custom_handler)

# Emit the event
hooks.emit(CustomHookName.CUSTOM_EVENT, {"data": "value"})
```

## Type Safety with Protocol Types

The Hooks system uses Python's `Protocol` types to provide better type safety for handler functions. This helps catch errors at development time and provides better IDE support with autocompletion.

If you're writing your own handlers, you can specify the appropriate type:

```python
from instructor.hooks import CompletionErrorHandler


def my_error_handler(error: Exception) -> None:
    print(f"Error occurred: {error}")


# Type checking will verify this is a valid error handler
handler: CompletionErrorHandler = my_error_handler

client.on("completion:error", handler)
```

## Error Handling with Hooks

Hooks provide an excellent way to monitor and handle errors consistently across your application. You can use them with Instructor's exception hierarchy for sophisticated error handling:

```python
from instructor.exceptions import (
    InstructorError,
    IncompleteOutputException,
    InstructorRetryException,
    ValidationError,
    ProviderError,
    ConfigurationError
)
import logging

logger = logging.getLogger(__name__)

class ErrorMonitor:
    def __init__(self):
        self.error_counts = {
            "incomplete": 0,
            "validation": 0,
            "provider": 0,
            "retry_exhausted": 0,
            "other": 0
        }

    def handle_error(self, error: Exception):
        # Log the error with appropriate level
        if isinstance(error, IncompleteOutputException):
            self.error_counts["incomplete"] += 1
            logger.warning(f"Incomplete output: {error}")
        elif isinstance(error, ValidationError):
            self.error_counts["validation"] += 1
            logger.error(f"Validation failed: {error}")
        elif isinstance(error, ProviderError):
            self.error_counts["provider"] += 1
            logger.error(f"Provider error ({error.provider}): {error}")
        elif isinstance(error, InstructorRetryException):
            self.error_counts["retry_exhausted"] += 1
            logger.critical(f"All retries exhausted after {error.n_attempts} attempts")
        else:
            self.error_counts["other"] += 1
            logger.error(f"Unexpected error: {type(error).__name__}: {error}")

    def get_stats(self):
        return self.error_counts

# Usage
monitor = ErrorMonitor()
client = instructor.from_provider("openai/gpt-4.1-mini")

client.on("completion:error", monitor.handle_error)
client.on("parse:error", monitor.handle_error)
client.on("completion:last_attempt", monitor.handle_error)

# After running your application
print(f"Error statistics: {monitor.get_stats()}")
```

## Hooks in Testing

Hooks are particularly useful for testing, as they allow you to inspect the arguments and responses without modifying your application code:

```python
import unittest
from unittest.mock import Mock
import instructor


class TestMyApp(unittest.TestCase):
    def test_completion(self):
        client = instructor.from_provider("openai/gpt-4.1-mini")
        mock_handler = Mock()

        client.on("completion:response", mock_handler)

        # Call your code that uses the client
        result = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            response_model=str,
        )

        # Verify the mock was called
        mock_handler.assert_called_once()

        # You can also inspect the arguments
        response_arg = mock_handler.call_args[0][0]
        self.assertEqual(response_arg.model, "gpt-4.1-mini")
```

This approach allows you to test your code without mocking the entire client.

### Using Hooks

```python
from openai import OpenAI
import instructor


# Initialize client
client = instructor.patch(OpenAI())

# Example with all hooks enabled (default)
response = client.chat.completions.create(
    response_model=str,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

```python
from enum import Enum, auto
import instructor
from openai import OpenAI


# Define standard hook names
class HookName(Enum):
    COMPLETION_KWARGS = auto()
    COMPLETION_RESPONSE = auto()
    COMPLETION_ERROR = auto()
    COMPLETION_LAST_ATTEMPT = auto()
    PARSE_ERROR = auto()


# Create a new enum for custom hooks
class CustomHookName(Enum):
    MY_CUSTOM_HOOK = "my_custom_hook"
    ANOTHER_HOOK = "another_hook"


# Initialize client with custom hooks
client = instructor.patch(OpenAI())
```
