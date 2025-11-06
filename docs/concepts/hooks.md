---
title: Understanding Hooks in the Instructor Library
description: Learn how to use hooks for event handling in the Instructor library to enhance logging, error handling, and custom behaviors with rich context objects and async support.
---

# Hooks

Hooks provide a powerful mechanism for intercepting and handling events during the completion and parsing process in the Instructor library. They allow you to add custom behavior, logging, error handling, and monitoring at various stages of the API interaction with rich contextual information.

## Overview

The Hooks system in Instructor is based on the `Hooks` class, which manages event registration and emission. It provides:

- **Rich Context Objects**: Every hook receives detailed metadata about the request
- **Async Support**: Hooks can be async functions for non-blocking operations
- **Priority System**: Control execution order when multiple hooks are registered
- **10 Lifecycle Events**: Cover the complete request lifecycle
- **Backward Compatibility**: Legacy hook signatures still work
- **Type Safety**: Protocol-based typing for better IDE support

## Quick Start

```python
import instructor
from instructor.core.hooks import Hooks, HookName, CompletionKwargsContext
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

client = instructor.from_openai(openai.OpenAI())

# Modern hook with context object
def log_request(ctx: CompletionKwargsContext) -> None:
    print(f"Request {ctx.context.request_id}")
    print(f"Attempt {ctx.context.attempt_number}/{ctx.context.total_attempts}")
    print(f"Model: {ctx.kwargs.get('model')}")
    print(f"Is retry: {ctx.context.is_retry}")

# Register with priority (higher = earlier execution)
client.on("completion:kwargs", log_request, priority=100)

# Make request - hook is automatically called
user = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Extract: John is 30"}],
    response_model=User,
)
```

---

## Hook Lifecycle

Hooks are emitted in a specific order during request processing:

```
1. request:start          → Before any processing
2. completion:kwargs      → Before API call (repeats on retry)
3. completion:response    → After successful API response
4. stream:chunk           → Each chunk in streaming mode
5. validation:success     → After successful validation
6. parse:error            → When validation fails
7. retry:attempt          → Before retry attempt
8. completion:error       → When API call fails
9. completion:last_attempt → Final retry failure
10. request:end           → After request completes
```

---

## Context Objects

All hooks receive rich context objects containing detailed metadata:

### HookContext (Base Context)

The base context included in all hook contexts:

```python
@dataclass
class HookContext:
    request_id: str              # Unique identifier for this request
    attempt_number: int          # Current retry attempt (1-indexed)
    total_attempts: int          # Maximum retry attempts
    is_retry: bool              # Whether this is a retry
    start_time: float           # Unix timestamp when request started
    mode: Mode                  # Mode being used (e.g., Mode.TOOLS)
    response_model: type | None # Expected Pydantic model class
```

### CompletionKwargsContext

Passed to `completion:kwargs` hook:

```python
@dataclass
class CompletionKwargsContext:
    context: HookContext          # Base context
    args: tuple[Any, ...]        # Positional args
    kwargs: dict[str, Any]       # Keyword args (model, messages, etc.)

# Usage example
def log_kwargs(ctx: CompletionKwargsContext) -> None:
    print(f"Request ID: {ctx.context.request_id}")
    print(f"Model: {ctx.kwargs.get('model')}")
    print(f"Messages: {ctx.kwargs.get('messages')}")
    print(f"Attempt: {ctx.context.attempt_number}")
```

### CompletionResponseContext

Passed to `completion:response` hook:

```python
@dataclass
class CompletionResponseContext:
    context: HookContext    # Base context
    response: Any          # Raw API response
    duration: float        # API call duration in seconds

# Usage example
def log_response(ctx: CompletionResponseContext) -> None:
    print(f"Duration: {ctx.duration:.2f}s")
    if hasattr(ctx.response, 'usage'):
        print(f"Tokens: {ctx.response.usage.total_tokens}")
```

### ErrorContext

Passed to `completion:error`, `parse:error`, and `completion:last_attempt` hooks:

```python
@dataclass
class ErrorContext:
    context: HookContext           # Base context
    error: Exception              # The exception that occurred
    kwargs: dict[str, Any]        # Request kwargs that failed
    response: Any | None          # Partial response if available
    failed_attempts: list[Any]    # All previous failures
    stack_trace: str              # Formatted traceback

# Usage example
def handle_error(ctx: ErrorContext) -> None:
    print(f"Error on attempt {ctx.context.attempt_number}")
    print(f"Error type: {type(ctx.error).__name__}")
    print(f"Message: {str(ctx.error)}")
    print(f"Previous failures: {len(ctx.failed_attempts)}")

    # Access the full stack trace
    if ctx.context.is_retry:
        print("This was a retry attempt")
        print(f"Stack trace:\n{ctx.stack_trace}")
```

### ValidationContext

Passed to `validation:success` hook:

```python
@dataclass
class ValidationContext:
    context: HookContext    # Base context
    response: Any          # Raw API response
    parsed_model: Any      # Successfully validated Pydantic model

# Usage example
def on_validation_success(ctx: ValidationContext) -> None:
    print(f"Successfully validated {type(ctx.parsed_model).__name__}")
    print(f"Model: {ctx.parsed_model}")
```

### RetryContext

Passed to `retry:attempt` hook:

```python
@dataclass
class RetryContext:
    context: HookContext         # Base context
    last_error: Exception       # Error that triggered retry
    next_kwargs: dict[str, Any] # Modified kwargs for next attempt

# Usage example
def on_retry(ctx: RetryContext) -> None:
    print(f"Retrying after: {type(ctx.last_error).__name__}")
    print(f"Next attempt: {ctx.context.attempt_number + 1}")
```

### StreamChunkContext

Passed to `stream:chunk` hook:

```python
@dataclass
class StreamChunkContext:
    context: HookContext  # Base context
    chunk: Any           # Streaming chunk data
    chunk_index: int     # Index in stream (0-indexed)

# Usage example
def on_chunk(ctx: StreamChunkContext) -> None:
    print(f"Chunk {ctx.chunk_index}: {ctx.chunk}")
```

---

## Hook Events Reference

### `request:start`

Emitted before any processing begins.

**Receives:** `HookContext`

```python
def on_request_start(ctx: HookContext) -> None:
    print(f"Starting request {ctx.request_id}")
    print(f"Mode: {ctx.mode}")
    print(f"Max attempts: {ctx.total_attempts}")

client.on("request:start", on_request_start)
```

### `completion:kwargs`

Emitted before each API call (repeats on retries).

**Receives:** `CompletionKwargsContext`

```python
def on_kwargs(ctx: CompletionKwargsContext) -> None:
    print(f"Calling {ctx.kwargs.get('model')}")
    if ctx.context.is_retry:
        print("This is a retry")

client.on("completion:kwargs", on_kwargs)
```

### `completion:response`

Emitted after successful API response.

**Receives:** `CompletionResponseContext`

```python
def on_response(ctx: CompletionResponseContext) -> None:
    print(f"Response received in {ctx.duration:.2f}s")
    print(f"Tokens: {ctx.response.usage.total_tokens}")

client.on("completion:response", on_response)
```

### `stream:chunk`

Emitted for each chunk in streaming mode.

**Receives:** `StreamChunkContext`

```python
def on_chunk(ctx: StreamChunkContext) -> None:
    print(f"Chunk {ctx.chunk_index}")

client.on("stream:chunk", on_chunk)
```

### `validation:success`

Emitted after successful Pydantic validation.

**Receives:** `ValidationContext`

```python
def on_valid(ctx: ValidationContext) -> None:
    print(f"Validated: {type(ctx.parsed_model).__name__}")

client.on("validation:success", on_valid)
```

### `parse:error`

Emitted when validation/parsing fails.

**Receives:** `ErrorContext`

```python
def on_parse_error(ctx: ErrorContext) -> None:
    print(f"Parse error: {ctx.error}")
    print(f"Failed attempts so far: {len(ctx.failed_attempts)}")

client.on("parse:error", on_parse_error)
```

### `retry:attempt`

Emitted before each retry attempt.

**Receives:** `RetryContext`

```python
def on_retry(ctx: RetryContext) -> None:
    print(f"Retrying after: {type(ctx.last_error).__name__}")

client.on("retry:attempt", on_retry)
```

### `completion:error`

Emitted when API call fails (network, API errors).

**Receives:** `ErrorContext`

```python
def on_api_error(ctx: ErrorContext) -> None:
    print(f"API error: {ctx.error}")

client.on("completion:error", on_api_error)
```

### `completion:last_attempt`

Emitted on final retry failure.

**Receives:** `ErrorContext`

```python
def on_last_attempt(ctx: ErrorContext) -> None:
    print(f"Final attempt failed after {ctx.context.attempt_number} tries")

client.on("completion:last_attempt", on_last_attempt)
```

### `request:end`

Emitted after request completes (success or failure).

**Receives:** `HookContext`

```python
def on_request_end(ctx: HookContext) -> None:
    duration = time() - ctx.start_time
    print(f"Request completed in {duration:.2f}s")

client.on("request:end", on_request_end)
```

---

## Async Hooks

Hooks can be async functions for non-blocking operations like logging to external services:

```python
import asyncio
import httpx

async def log_to_datadog(ctx: CompletionKwargsContext) -> None:
    """Async hook for external logging"""
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.datadoghq.com/api/v1/logs",
            json={
                "request_id": ctx.context.request_id,
                "model": ctx.kwargs.get("model"),
                "attempt": ctx.context.attempt_number,
            }
        )

async def track_metrics(ctx: CompletionResponseContext) -> None:
    """Async metrics tracking"""
    await metrics_service.record({
        "duration": ctx.duration,
        "tokens": ctx.response.usage.total_tokens,
    })

# Register async hooks (works with both sync and async clients)
client.on("completion:kwargs", log_to_datadog, priority=100)
client.on("completion:response", track_metrics)

# Async hooks are awaited if the client is async, or run in sync context
```

**Key Points:**
- Async hooks work with both sync and async Instructor clients
- With sync clients, async hooks are executed using `asyncio.run()`
- With async clients, hooks are properly awaited
- Hook errors don't crash the request (logged as warnings)

---

## Priority System

Control execution order when multiple hooks are registered for the same event:

```python
# Higher priority = executes first (default is 0)

def auth_check(ctx: CompletionKwargsContext) -> None:
    print("1. Auth check (priority 100)")
    # Verify API key is present

def log_request(ctx: CompletionKwargsContext) -> None:
    print("2. Logging (priority 50)")
    # Log the request

def track_metrics(ctx: CompletionKwargsContext) -> None:
    print("3. Metrics (priority 0)")
    # Track metrics

# Register with priorities
client.on("completion:kwargs", auth_check, priority=100)
client.on("completion:kwargs", log_request, priority=50)
client.on("completion:kwargs", track_metrics, priority=0)

# Execution order: auth_check → log_request → track_metrics
```

**Use Cases for Priorities:**
- Authentication/validation hooks execute first
- Logging happens before metrics
- Cleanup hooks execute last
- Critical hooks guaranteed to run before others

---

## Backward Compatibility

Old-style hooks without context objects still work:

```python
# Old style (still supported)
def old_kwargs_hook(*args, **kwargs) -> None:
    print(f"Model: {kwargs.get('model')}")

def old_response_hook(response) -> None:
    print(f"Response: {response}")

def old_error_hook(error: Exception) -> None:
    print(f"Error: {error}")

# These still work!
client.on("completion:kwargs", old_kwargs_hook)
client.on("completion:response", old_response_hook)
client.on("completion:error", old_error_hook)
```

**Migration Strategy:**

```python
# Before (old style)
def log_completion(*args, **kwargs):
    print(kwargs.get('model'))

# After (new style with context)
def log_completion(ctx: CompletionKwargsContext):
    print(f"Request {ctx.context.request_id}")
    print(f"Model: {ctx.kwargs.get('model')}")
    print(f"Attempt: {ctx.context.attempt_number}")
```

The library automatically detects handler signatures and calls them appropriately.

---

## Common Patterns

### Request Timing

```python
from time import time

request_times = {}

def on_start(ctx: HookContext) -> None:
    request_times[ctx.request_id] = time()

def on_end(ctx: HookContext) -> None:
    duration = time() - request_times[ctx.request_id]
    print(f"Total duration: {duration:.2f}s")
    del request_times[ctx.request_id]

client.on("request:start", on_start)
client.on("request:end", on_end)
```

### Token Usage Tracking

```python
class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.request_count = 0

    def track(self, ctx: CompletionResponseContext) -> None:
        if hasattr(ctx.response, 'usage'):
            tokens = ctx.response.usage.total_tokens
            self.total_tokens += tokens
            self.request_count += 1
            print(f"Request {ctx.context.request_id}: {tokens} tokens")
            print(f"Total so far: {self.total_tokens} across {self.request_count} requests")

tracker = TokenTracker()
client.on("completion:response", tracker.track)
```

### Error Rate Monitoring

```python
class ErrorMonitor:
    def __init__(self):
        self.errors = []

    def track_error(self, ctx: ErrorContext) -> None:
        self.errors.append({
            "request_id": ctx.context.request_id,
            "attempt": ctx.context.attempt_number,
            "error_type": type(ctx.error).__name__,
            "error_msg": str(ctx.error),
            "timestamp": ctx.context.start_time,
        })

        # Alert if error rate exceeds threshold
        recent_errors = [
            e for e in self.errors
            if time() - e["timestamp"] < 300  # Last 5 minutes
        ]
        if len(recent_errors) > 10:
            print("⚠️ HIGH ERROR RATE DETECTED!")

    def get_stats(self):
        return {
            "total_errors": len(self.errors),
            "error_types": Counter(e["error_type"] for e in self.errors),
        }

monitor = ErrorMonitor()
client.on("parse:error", monitor.track_error)
client.on("completion:error", monitor.track_error)
```

### Retry Logging with Context

```python
def log_retry(ctx: RetryContext) -> None:
    print(f"🔄 Retry {ctx.context.attempt_number}/{ctx.context.total_attempts}")
    print(f"   Last error: {type(ctx.last_error).__name__}")
    print(f"   Request ID: {ctx.context.request_id}")

    # Log modified kwargs
    if ctx.next_kwargs != ctx.context:  # Check if kwargs were modified
        print(f"   Modified kwargs for retry")

client.on("retry:attempt", log_retry)
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, threshold=5, timeout=60):
        self.failures = 0
        self.threshold = threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.is_open = False

    def on_error(self, ctx: ErrorContext) -> None:
        self.failures += 1
        self.last_failure_time = time()

        if self.failures >= self.threshold:
            self.is_open = True
            print(f"🔴 Circuit breaker OPEN after {self.failures} failures")

    def on_success(self, ctx: ValidationContext) -> None:
        # Reset on success
        if self.is_open:
            print("🟢 Circuit breaker CLOSED")
        self.failures = 0
        self.is_open = False

    def check_circuit(self, ctx: CompletionKwargsContext) -> None:
        if self.is_open:
            # Check if timeout has passed
            if time() - self.last_failure_time > self.timeout:
                print("⚡ Circuit breaker half-open (testing)")
                self.failures = self.threshold - 1  # Allow one attempt
            else:
                raise Exception("Circuit breaker is OPEN")

breaker = CircuitBreaker()
client.on("completion:kwargs", breaker.check_circuit, priority=100)
client.on("completion:error", breaker.on_error)
client.on("validation:success", breaker.on_success)
```

### Distributed Tracing

```python
import uuid
from contextvars import ContextVar

# Use context vars for request correlation
trace_context = ContextVar("trace_context", default=None)

def start_trace(ctx: HookContext) -> None:
    """Initialize distributed trace"""
    trace_id = str(uuid.uuid4())
    trace_context.set({
        "trace_id": trace_id,
        "request_id": ctx.request_id,
        "start_time": ctx.start_time,
    })
    print(f"🔍 Trace started: {trace_id}")

def log_with_trace(ctx: CompletionKwargsContext) -> None:
    """Log with trace context"""
    trace = trace_context.get()
    print(f"[Trace: {trace['trace_id']}] Calling API")

def end_trace(ctx: HookContext) -> None:
    """End distributed trace"""
    trace = trace_context.get()
    duration = time() - trace["start_time"]
    print(f"🔍 Trace ended: {trace['trace_id']} ({duration:.2f}s)")
    trace_context.set(None)

client.on("request:start", start_trace)
client.on("completion:kwargs", log_with_trace)
client.on("request:end", end_trace)
```

---

## Hook Combination

Compose different hook sets for different use cases:

### Basic Combination

```python
from instructor.core.hooks import Hooks

# Create separate hook sets
logging_hooks = Hooks()
logging_hooks.on("completion:kwargs", log_request)
logging_hooks.on("completion:response", log_response)

metrics_hooks = Hooks()
metrics_hooks.on("completion:response", track_tokens)

# Combine using + operator
combined = logging_hooks + metrics_hooks
client = instructor.from_openai(openai.OpenAI(), hooks=combined)

# Or combine multiple at once
all_hooks = Hooks.combine(logging_hooks, metrics_hooks, debug_hooks)
```

### Per-Call Hooks

Override or extend hooks for specific requests:

```python
# Client with standard hooks
client = instructor.from_openai(openai.OpenAI())
client.on("completion:kwargs", standard_logging)

# Create special hooks for specific call
debug_hooks = Hooks()
debug_hooks.on("parse:error", detailed_error_logging)
debug_hooks.on("completion:response", response_inspection)

# Per-call hooks are combined with client hooks
user = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Extract: Complex data"}],
    response_model=User,
    hooks=debug_hooks  # Combined with client hooks
)
```

---

## Hook Management

### Registering Hooks

```python
# Method 1: Using enum
from instructor.core.hooks import HookName
client.on(HookName.COMPLETION_KWARGS, handler)

# Method 2: Using string
client.on("completion:kwargs", handler)

# Method 3: With priority
client.on("completion:kwargs", handler, priority=100)
```

### Removing Hooks

```python
# Remove specific handler
client.off("completion:kwargs", handler)

# Clear all handlers for one event
client.clear("completion:kwargs")

# Clear all hooks
client.clear()
```

### Copying Hooks

```python
# Create independent copy
original_hooks = Hooks()
original_hooks.on("completion:kwargs", handler)

copy_hooks = original_hooks.copy()
# Modifications to copy don't affect original
```

---

## Type Safety

Use Protocol types for better IDE support:

```python
from instructor.core.hooks import (
    CompletionKwargsHandler,
    CompletionResponseHandler,
    ErrorHandler,
)

# Type-checked handlers
def my_kwargs_handler(ctx: CompletionKwargsContext) -> None:
    print(ctx.kwargs)

def my_response_handler(ctx: CompletionResponseContext) -> None:
    print(ctx.response)

def my_error_handler(ctx: ErrorContext) -> None:
    print(ctx.error)

# Type checking verifies signatures
handler1: CompletionKwargsHandler = my_kwargs_handler
handler2: CompletionResponseHandler = my_response_handler
handler3: ErrorHandler = my_error_handler

client.on("completion:kwargs", handler1)
client.on("completion:response", handler2)
client.on("completion:error", handler3)
```

---

## Testing with Hooks

Hooks are excellent for testing:

```python
import pytest
from unittest.mock import Mock

def test_request_hooks():
    client = instructor.from_openai(openai.OpenAI())

    # Create mock handlers
    kwargs_mock = Mock()
    response_mock = Mock()

    client.on("completion:kwargs", kwargs_mock)
    client.on("completion:response", response_mock)

    # Make request
    result = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        response_model=str,
    )

    # Verify hooks were called
    kwargs_mock.assert_called_once()
    response_mock.assert_called_once()

    # Inspect arguments
    ctx = kwargs_mock.call_args[0][0]
    assert isinstance(ctx, CompletionKwargsContext)
    assert ctx.kwargs["model"] == "gpt-4"
    assert ctx.context.attempt_number == 1

def test_error_handling():
    client = instructor.from_openai(openai.OpenAI())

    error_mock = Mock()
    client.on("parse:error", error_mock)

    # Trigger validation error
    with pytest.raises(ValidationError):
        client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Invalid"}],
            response_model=User,
            max_retries=1,
        )

    # Verify error hook was called
    assert error_mock.called

    # Check error context
    ctx = error_mock.call_args[0][0]
    assert isinstance(ctx, ErrorContext)
    assert isinstance(ctx.error, ValidationError)
```

---

## Best Practices

1. **Use Context Objects**: Access rich metadata for better debugging and monitoring
2. **Set Priorities**: Ensure critical hooks execute first (auth, validation)
3. **Async for I/O**: Use async hooks for external API calls (logging services, metrics)
4. **Error Handling**: Remember hook errors are caught and logged, don't crash requests
5. **Clean Up**: Use `request:end` hook for cleanup (close connections, delete temp data)
6. **Test Your Hooks**: Mock hooks in tests to verify they're called correctly
7. **Combine Hooks**: Create reusable hook sets for different environments (dev, prod)
8. **Type Safety**: Use Protocol types for better IDE support and type checking

---

## Comprehensive Example

```python
import instructor
import openai
from instructor.core.hooks import *
from pydantic import BaseModel
from time import time
import asyncio

class User(BaseModel):
    name: str
    age: int

# Monitoring class with multiple hook handlers
class RequestMonitor:
    def __init__(self):
        self.requests = {}
        self.total_tokens = 0

    def on_start(self, ctx: HookContext) -> None:
        self.requests[ctx.request_id] = {
            "start_time": ctx.start_time,
            "attempt": ctx.attempt_number,
            "mode": ctx.mode,
        }
        print(f"🚀 Request started: {ctx.request_id}")

    def on_kwargs(self, ctx: CompletionKwargsContext) -> None:
        req = self.requests[ctx.context.request_id]
        req["model"] = ctx.kwargs.get("model")
        print(f"📤 Calling {req['model']} (attempt {ctx.context.attempt_number})")

    def on_response(self, ctx: CompletionResponseContext) -> None:
        if hasattr(ctx.response, 'usage'):
            tokens = ctx.response.usage.total_tokens
            self.total_tokens += tokens
            self.requests[ctx.context.request_id]["tokens"] = tokens
            print(f"📊 Used {tokens} tokens ({ctx.duration:.2f}s)")

    def on_validation(self, ctx: ValidationContext) -> None:
        print(f"✅ Validated: {type(ctx.parsed_model).__name__}")

    def on_error(self, ctx: ErrorContext) -> None:
        print(f"❌ Error: {type(ctx.error).__name__} on attempt {ctx.context.attempt_number}")
        if ctx.context.attempt_number < ctx.context.total_attempts:
            print(f"   Will retry ({ctx.context.total_attempts - ctx.context.attempt_number} left)")

    def on_retry(self, ctx: RetryContext) -> None:
        print(f"🔄 Retrying after {type(ctx.last_error).__name__}")

    def on_end(self, ctx: HookContext) -> None:
        req = self.requests.pop(ctx.request_id)
        duration = time() - req["start_time"]
        print(f"🏁 Request completed in {duration:.2f}s")
        print(f"📈 Total tokens used: {self.total_tokens}")

# Initialize client with monitoring
client = instructor.from_openai(openai.OpenAI())
monitor = RequestMonitor()

# Register all hooks with appropriate priorities
client.on("request:start", monitor.on_start, priority=100)
client.on("completion:kwargs", monitor.on_kwargs, priority=50)
client.on("completion:response", monitor.on_response)
client.on("validation:success", monitor.on_validation)
client.on("parse:error", monitor.on_error)
client.on("completion:error", monitor.on_error)
client.on("retry:attempt", monitor.on_retry)
client.on("request:end", monitor.on_end, priority=-100)  # Run last

# Make request - all hooks automatically called
user = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Extract: Alice is 30"}],
    response_model=User,
    max_retries=3,
)

print(f"\nResult: {user}")
```

---

## Advanced: Custom Hook Events

While Instructor provides 10 built-in hooks, you can emit custom events:

```python
from instructor.core.hooks import Hooks, HookName

hooks = Hooks()

# Define custom handler
def custom_handler(data):
    print(f"Custom event: {data}")

# Register for custom event (works with any string)
hooks.on("custom:event", custom_handler)

# Emit custom event
hooks.emit(HookName("custom:event"), {"data": "value"})
```

---

## API Reference

### Hooks Class

```python
class Hooks:
    def __init__(self) -> None
    def on(hook_name: HookNameType, handler: HandlerType, priority: int = 0) -> None
    def off(hook_name: HookNameType, handler: HandlerType) -> None
    def clear(hook_name: HookNameType | None = None) -> None
    def copy() -> Hooks
    def __add__(other: Hooks) -> Hooks
    def __iadd__(other: Hooks) -> Hooks
    @classmethod
    def combine(*hooks_instances: Hooks) -> Hooks
```

### Instructor Client Methods

```python
client.on(hook_name, handler, priority=0)  # Register hook
client.off(hook_name, handler)              # Remove hook
client.clear(hook_name=None)                # Clear hooks
```

### Hook Events

```python
"request:start"             # HookContext
"request:end"               # HookContext
"completion:kwargs"         # CompletionKwargsContext
"completion:response"       # CompletionResponseContext
"completion:error"          # ErrorContext
"completion:last_attempt"   # ErrorContext
"parse:error"               # ErrorContext
"validation:success"        # ValidationContext
"retry:attempt"             # RetryContext
"stream:chunk"              # StreamChunkContext
```

---

## Migration from Old API

If you're using the old hook API, here's how to migrate:

```python
# OLD API
def old_handler(*args, **kwargs):
    model = kwargs.get('model')
    print(f"Model: {model}")

client.on("completion:kwargs", old_handler)

# NEW API
def new_handler(ctx: CompletionKwargsContext):
    model = ctx.kwargs.get('model')
    request_id = ctx.context.request_id
    attempt = ctx.context.attempt_number
    print(f"Request {request_id}, attempt {attempt}: {model}")

client.on("completion:kwargs", new_handler)
```

**Benefits of Migration:**
- Access to request_id for correlation
- Retry attempt information
- Timing data (start_time)
- Mode and response_model info
- Full error context including stack traces
- Type safety with Protocol types

**Note:** Old-style handlers continue to work for backward compatibility.

---

## Conclusion

Hooks in Instructor provide a powerful, flexible system for monitoring, logging, and controlling LLM interactions. With rich context objects, async support, priorities, and comprehensive lifecycle coverage, you can build sophisticated observability and control systems on top of your LLM applications.

Key takeaways:
- Use context objects for rich metadata
- Leverage async hooks for non-blocking I/O
- Set priorities for critical hooks
- Combine hooks for reusable monitoring sets
- Test hooks thoroughly
- Migrate to new API for better features

For more examples, see the [examples/hooks](https://github.com/jxnl/instructor/tree/main/examples/hooks) directory.
