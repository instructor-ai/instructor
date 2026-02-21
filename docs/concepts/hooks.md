---
title: Hooks
description: Learn how to use hooks for event handling, logging, and error handling in Instructor.
---

# Hooks

Hooks let you intercept and handle events during the completion and parsing process. Use them to add logging, monitoring, or error handling at different stages of API interactions.

## Hook Events

| Event | Description | Handler Signature |
|-------|-------------|-------------------|
| `completion:kwargs` | Arguments passed to completion | `def handler(*args, **kwargs)` |
| `completion:response` | Raw API response received | `def handler(response)` |
| `completion:error` | Error before retries | `def handler(error)` |
| `parse:error` | Pydantic validation failed | `def handler(error)` |
| `completion:last_attempt` | Last retry attempt | `def handler(error)` |

## Registering and Removing Hooks

```python
import instructor

client = instructor.from_provider("openai/gpt-4.1-mini")


def log_kwargs(*args, **kwargs):
    print(f"Model: {kwargs.get('model')}")


def log_response(response):
    print(f"Response received: {response.id}")


# Register hooks
client.on("completion:kwargs", log_kwargs)
client.on("completion:response", log_response)

# Make a request
resp = client.create(
    messages=[{"role": "user", "content": "Hello, world!"}],
    response_model=str,
)

# Remove a specific hook
client.off("completion:kwargs", log_kwargs)

# Clear all hooks for an event
client.clear("completion:kwargs")

# Clear all hooks
client.clear()
```

You can use enum values or strings for hook names:

```python
from instructor.hooks import HookName

client.on(HookName.COMPLETION_KWARGS, log_kwargs)  # Using enum
client.on("completion:kwargs", log_kwargs)          # Using string
```

## Practical Example: Logging

```python
import instructor
from pydantic import BaseModel


class ErrorCounter:
    def __init__(self):
        self.count = 0

    def handle_error(self, error: Exception):
        self.count += 1
        print(f"Error #{self.count}: {type(error).__name__}: {error}")


client = instructor.from_provider("openai/gpt-4.1-mini")
counter = ErrorCounter()

client.on("completion:error", counter.handle_error)
client.on("parse:error", counter.handle_error)


class User(BaseModel):
    name: str
    age: int


try:
    user = client.create(
        messages=[{"role": "user", "content": "Extract: John is twenty"}],
        response_model=User,
    )
    print(f"Extracted: {user}")
except Exception as e:
    print(f"Final error: {e}")

print(f"Total errors: {counter.count}")
```

## Error Handling

Monitor errors by type using Instructor's exception hierarchy:

```python
import logging
import instructor
from instructor.core.exceptions import (
    IncompleteOutputException,
    InstructorRetryException,
    ValidationError,
    ProviderError,
)

logger = logging.getLogger(__name__)


def handle_error(error: Exception):
    if isinstance(error, IncompleteOutputException):
        logger.warning(f"Incomplete output: {error}")
    elif isinstance(error, ValidationError):
        logger.error(f"Validation failed: {error}")
    elif isinstance(error, ProviderError):
        logger.error(f"Provider error ({error.provider}): {error}")
    elif isinstance(error, InstructorRetryException):
        logger.critical(f"Retries exhausted after {error.n_attempts} attempts")
    else:
        logger.error(f"Unexpected error: {error}")


client = instructor.from_provider("openai/gpt-4.1-mini")
client.on("completion:error", handle_error)
client.on("parse:error", handle_error)
```

## Hook Combination

Combine different hook sets using the `+` operator:

```python
import instructor
from instructor.core.hooks import Hooks

# Create specialized hook sets
logging_hooks = Hooks()
logging_hooks.on("completion:kwargs", lambda **kw: print("Logging kwargs"))

metrics_hooks = Hooks()
metrics_hooks.on("completion:response", lambda resp: print("Recording metrics"))

# Combine hooks
combined = logging_hooks + metrics_hooks

# Or combine multiple at once
all_hooks = Hooks.combine(logging_hooks, metrics_hooks)

client = instructor.from_provider("openai/gpt-4.1-mini", hooks=combined)
```

## Per-Call Hooks

Specify hooks for individual API calls:

```python
import instructor
from instructor.core.hooks import Hooks
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Client with standard hooks
client_hooks = Hooks()
client_hooks.on("completion:kwargs", lambda **kw: print("Standard logging"))

client = instructor.from_provider("openai/gpt-4.1-mini", hooks=client_hooks)

# Debug hooks for specific calls
debug_hooks = Hooks()
debug_hooks.on("parse:error", lambda err: print(f"Debug: {err}"))

# Per-call hooks combine with client hooks
user = client.create(
    messages=[{"role": "user", "content": "Extract: Alice is 25"}],
    response_model=User,
    hooks=debug_hooks,  # Both client and debug hooks run
)
```

## Testing with Hooks

Use hooks to inspect requests and responses in tests:

```python
import unittest
from unittest.mock import Mock
import instructor


class TestMyApp(unittest.TestCase):
    def test_completion(self):
        client = instructor.from_provider("openai/gpt-4.1-mini")
        mock_handler = Mock()

        client.on("completion:response", mock_handler)

        result = client.create(
            messages=[{"role": "user", "content": "Hello"}],
            response_model=str,
        )

        mock_handler.assert_called_once()
        response = mock_handler.call_args[0][0]
        self.assertEqual(response.model, "gpt-4.1-mini")
```

## Custom Hooks

Create custom hook systems by extending the base pattern:

```python
from enum import Enum
from instructor.hooks import HookName


class CustomHookName(str, Enum):
    CUSTOM_EVENT = "custom:event"
    # Include base hooks for compatibility
    COMPLETION_KWARGS = HookName.COMPLETION_KWARGS.value


class CustomHooks:
    def __init__(self):
        self._handlers: dict[str, list] = {}

    def on(self, hook_name: CustomHookName, handler):
        self._handlers.setdefault(hook_name.value, []).append(handler)

    def emit(self, hook_name: CustomHookName, payload):
        for handler in self._handlers.get(hook_name.value, []):
            handler(payload)


hooks = CustomHooks()
hooks.on(CustomHookName.CUSTOM_EVENT, lambda data: print(f"Custom: {data}"))
hooks.emit(CustomHookName.CUSTOM_EVENT, {"key": "value"})
```

## See Also

- [Debugging](../debugging.md) - Practical debugging techniques
- [Retrying](./retrying.md) - Monitor retry attempts
- [Error Handling](./error_handling.md) - Exception handling patterns
