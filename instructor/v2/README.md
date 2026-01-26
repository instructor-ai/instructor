# V2 Core Architecture

This document covers the v2 core infrastructure, including the registry-based design, exception handling, and component interactions.

## Overview

The v2 architecture uses a hierarchical registry system for managing provider modes and their corresponding handlers. It replaces the monolithic v1 approach with modular, composable components:

- **Registry**: Central mode/handler management
- **Handlers**: Pluggable request/response/reask handlers per mode
- **Patch**: Unified function patching mechanism
- **Retry**: Intelligent retry with registry-based handling
- **Exceptions**: Organized, centralized error handling

## Core Components

### Protocols (`instructor/v2/core/protocols.py`)

Type-safe interfaces for handlers:

- `RequestHandler` - Prepares request kwargs for a mode
- `ResponseParser` - Parses API response into Pydantic model
- `ReaskHandler` - Handles validation failures for retry

### Mode Registry (`instructor/v2/core/registry.py`)

The mode registry manages all available modes for each provider. It maps `(Provider, Mode)` tuples to their handler implementations.

**Key Features**:

- Provider/mode combination lookup
- Handler registration and retrieval
- Mode listing and discovery
- Fast O(1) lookups for handler dispatch

**Registry API**:

```python
from instructor.v2.core.registry import mode_registry
from instructor import Provider, Mode

# Get handlers (preferred)
handlers = mode_registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)

# Query
modes = mode_registry.get_modes_for_provider(Provider.ANTHROPIC)
is_registered = mode_registry.is_registered(Provider.ANTHROPIC, Mode.TOOLS)
```

Handlers are registered via `@register_mode_handler` decorator (see Handler Registration).

### Patch Mechanism (`instructor/v2/core/patch.py`)

Wraps provider API functions to add structured output support. Auto-detects sync/async, validates mode registration, injects default models, and integrates with registry handlers.

```python
from instructor.v2.core.patch import patch_v2

patched_create = patch_v2(
    client.messages.create,
    provider=Provider.ANTHROPIC,
    mode=Mode.TOOLS,
    default_model="claude-3-5-sonnet-20241022"
)
```

### Retry Logic (`instructor/v2/core/retry.py`)

Handles retries with registry-based reask logic. On `ValidationError`, uses registry handlers to generate reask prompts and retries up to `max_retries` times.

## Exception Handling

V2 exceptions inherit from `instructor.core.exceptions.InstructorError`:

- `RegistryError` - Mode not registered or handler lookup failure
- `ValidationContextError` - Conflicting `context`/`validation_context` parameters
- `InstructorRetryException` - Max retries exceeded with full attempt context

`RegistryValidationMixin` provides validation utilities used internally.

## Handler System

Handlers are pluggable components that implement provider-specific logic. They can be implemented as classes (using `ModeHandler` ABC) or as standalone functions (using Protocols).

### Handler Base Class (`instructor/v2/core/handler.py`)

The `ModeHandler` abstract base class provides a structured way to implement handlers:

```python
from instructor.v2.core.handler import ModeHandler
from pydantic import BaseModel
from typing import Any

class MyModeHandler(ModeHandler):
    """Handler for a specific mode."""

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request kwargs for this mode."""
        # Modify kwargs for mode-specific requirements
        return response_model, kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle validation failure and prepare retry."""
        # Modify kwargs for retry attempt
        return kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Parse API response into validated Pydantic model."""
        # Extract and validate response
        return response_model.model_validate(...)
```

### Handler Registration

All handlers must be registered using the `@register_mode_handler` decorator. This is the **only supported way** to register handlers in v2.

```python
from instructor.v2.core.decorators import register_mode_handler
from instructor import Provider, Mode
from instructor.v2.core.handler import ModeHandler

@register_mode_handler(Provider.ANTHROPIC, Mode.TOOLS)
class AnthropicToolsHandler(ModeHandler):
    """Handler automatically registered on import.

    The decorator internally calls mode_registry.register() with the
    handler methods mapped to the protocol functions.
    """

    def prepare_request(self, response_model, kwargs):
        # Implementation
        return response_model, kwargs

    def handle_reask(self, kwargs, response, exception):
        # Implementation
        return kwargs

    def parse_response(self, response, response_model, **kwargs):
        # Implementation
        return response_model.model_validate(...)
```

**How it works**: The decorator instantiates the handler class and calls `mode_registry.register()` with the handler's methods mapped to the protocol functions:

- `handler.prepare_request` → `request_handler`
- `handler.handle_reask` → `reask_handler`
- `handler.parse_response` → `response_parser`

**Benefits**:

- Automatic registration on import (no manual calls needed)
- Clean, declarative syntax
- Type-safe and consistent with the codebase pattern
- Used by all v2 providers (see `instructor/v2/providers/anthropic/handlers.py`)

**Important**: Direct calls to `mode_registry.register()` are not supported. All handlers must use the `@register_mode_handler` decorator.

## Execution Flow

### Sync Execution Path

```text
Client.create() with response_model
  ↓
patch_v2() [registry validation]
  ↓
new_create_sync()
  ├─ handle_context() [parameter validation]
  └─ retry_sync_v2() [retry logic]
      ├─ validate_mode_registration()
      ├─ For each attempt:
      │  ├─ Call original API
      │  ├─ Get handlers from registry
      │  ├─ Parse response via handler
      │  ├─ On success → return
      │  └─ On ValidationError:
      │     ├─ Record attempt
      │     ├─ Get reask via handler
      │     └─ Retry
      └─ Max retries exceeded → InstructorRetryException
```

### Async Execution Path

```text
AsyncClient.create() with response_model
  ↓
patch_v2() [registry validation]
  ↓
new_create_async()
  ├─ handle_context() [parameter validation]
  └─ retry_async_v2() [async retry logic]
      ├─ validate_mode_registration()
      ├─ For each attempt:
      │  ├─ Await API call
      │  ├─ Get handlers from registry
      │  ├─ Parse response via handler
      │  ├─ On success → return
      │  └─ On ValidationError:
      │     ├─ Record attempt
      │     ├─ Get reask via handler
      │     └─ Retry
      └─ Max retries exceeded → InstructorRetryException
```

## Error Handling Strategy

- **Fail fast**: Mode validation at patch time
- **Context validation**: `context`/`validation_context` conflict detection
- **Comprehensive logging**: All stages logged with attempt numbers
- **Exception chaining**: Full context preserved in exception chain

## Configuration

- **Mode**: Specified when creating client (`from_anthropic(client, mode=Mode.TOOLS)`)
- **Default Model**: Injected via `patch_v2(..., default_model="...")` if not provided in request
- **Max Retries**: Per-request via `max_retries=3` or `Retrying(...)` instance

## Adding a New Provider

1. **Add Provider Enum** (`instructor/utils.py`):

```python
class Provider(Enum):
    YOUR_PROVIDER = "your_provider"
```

2. **Create Handler** (`instructor/v2/providers/your_provider/handlers.py`):

```python
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.decorators import register_mode_handler
from instructor import Provider, Mode

@register_mode_handler(Provider.YOUR_PROVIDER, Mode.TOOLS)
class YourProviderToolsHandler(ModeHandler):
    def prepare_request(self, response_model, kwargs):
        # Convert response_model to provider tools format
        return response_model, kwargs

    def parse_response(self, response, response_model, **kwargs):
        # Extract and validate response
        return response_model.model_validate(...)

    def handle_reask(self, kwargs, response, exception):
        # Add error message for retry
        return kwargs
```

3. **Create Factory** (`instructor/v2/providers/your_provider/client.py`):

```python
from instructor.v2.providers.your_provider import handlers  # noqa: F401
from instructor.v2.core.patch import patch_v2
from instructor import Instructor, AsyncInstructor, Mode, Provider

@overload
def from_your_provider(client: YourProviderClient, mode=Mode.TOOLS) -> Instructor: ...

def from_your_provider(client, mode=Mode.TOOLS):
    patched_create = patch_v2(
        client.messages.create,
        provider=Provider.YOUR_PROVIDER,
        mode=mode,
    )
    return Instructor(client=client, create=patched_create, mode=mode)
```

4. **Export** (`instructor/v2/providers/your_provider/__init__.py`):

```python
from . import handlers  # noqa: F401
from .client import from_your_provider
__all__ = ["from_your_provider"]
```

See `instructor/v2/providers/anthropic/` for a complete example.

## Comprehensive Migration Guide: V1 to V2

This guide walks through migrating a provider from v1 to v2 architecture.

### Understanding V1 vs V2 Architecture

**V1 Architecture**:

- Mode-specific logic scattered across `process_response()` and utility functions
- Direct function calls based on mode checks
- Runtime mode validation
- Provider-specific mode enums (e.g., `ANTHROPIC_TOOLS`, `COHERE_JSON_SCHEMA`)

**V2 Architecture**:

- Centralized registry-based handler system
- Pluggable handlers per provider/mode combination
- Compile-time mode validation
- Generic mode enums with normalization (e.g., `TOOLS`, `JSON`)

### Step-by-Step Migration Process

#### Step 1: Analyze Your V1 Implementation

Before migrating, understand your current v1 provider:

1. **Locate provider files**:
   - `instructor/providers/your_provider/client.py` - Factory function
   - `instructor/providers/your_provider/utils.py` - Helper functions (if exists)

2. **Identify key components**:
   - What modes does your provider support?
   - What's the main API function being patched? (e.g., `client.chat`, `client.messages.create`)
   - How does request preparation work? (converting `response_model` to provider format)
   - How does response parsing work? (extracting structured data from raw response)
   - How does reask/retry work? (handling validation failures)

3. **Example V1 structure** (from `instructor/providers/cohere/client.py`):

```python
# V1: Direct mode handling in factory function
def from_cohere(client, mode=Mode.COHERE_TOOLS, **kwargs):
    valid_modes = {Mode.COHERE_TOOLS, Mode.COHERE_JSON_SCHEMA}

    if mode not in valid_modes:
        raise ModeError(...)

    # Uses instructor.patch() which routes to v1 process_response()
    return Instructor(
        client=client,
        create=instructor.patch(create=client.chat, mode=mode),
        provider=Provider.COHERE,
        mode=mode,
        **kwargs,
    )
```

#### Step 2: Create V2 Provider Directory Structure

Create the v2 provider directory:

```bash
mkdir -p instructor/v2/providers/your_provider
touch instructor/v2/providers/your_provider/__init__.py
touch instructor/v2/providers/your_provider/client.py
touch instructor/v2/providers/your_provider/handlers.py
```

#### Step 3: Map V1 Modes to V2 Modes

Determine which generic v2 modes your provider supports:

- `Mode.TOOLS` - Function calling / tool use
- `Mode.JSON` - JSON mode with schema instructions
- `Mode.JSON_SCHEMA` - Native structured outputs (if supported)
- `Mode.PARALLEL_TOOLS` - Parallel tool calling (if supported)

**Mode Normalization**: Provider-specific modes (e.g., `COHERE_TOOLS`) map to generic modes (`TOOLS`) via `normalize_mode()`.

#### Step 4: Extract Handler Logic from V1

Identify the three handler methods needed:

1. **Request Preparation** (`prepare_request`):
   - Look for functions like `handle_cohere_modes()`, `handle_anthropic_json()`
   - These convert `response_model` to provider-specific format
   - Modify request kwargs (e.g., add `tools` parameter)

2. **Response Parsing** (`parse_response`):
   - Look for functions in `process_response()` or `utils.py`
   - Extract structured data from raw API response
   - Validate against `response_model` using Pydantic

3. **Reask Handling** (`handle_reask`):
   - Look for functions like `reask_cohere_tools()`, `reask_anthropic_json()`
   - Modify kwargs to include error context for retry

**Example V1 handler functions** (from `instructor/providers/cohere/utils.py`):

```python
# V1: Scattered handler functions
def handle_cohere_json_schema(response_model, new_kwargs):
    # Convert response_model to JSON schema format
    new_kwargs["response_format"] = {"type": "json_schema", ...}
    return response_model, new_kwargs

def reask_cohere_tools(kwargs, response, exception):
    # Add error message to messages for retry
    kwargs["messages"].append({"role": "user", "content": f"Error: {exception}"})
    return kwargs
```

#### Step 5: Implement V2 Handlers

Create handler classes using the `@register_mode_handler` decorator:

```python
# instructor/v2/providers/your_provider/handlers.py
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.decorators import register_mode_handler
from instructor import Provider, Mode
from pydantic import BaseModel
from typing import Any

@register_mode_handler(Provider.COHERE, Mode.TOOLS)
class CohereToolsHandler(ModeHandler):
    """Handler for Cohere TOOLS mode."""

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Convert response_model to Cohere tools format."""
        if response_model is None:
            return None, kwargs

        # Convert response_model to Cohere function/tool format
        # (extract logic from v1 handle_cohere_modes)
        tool_schema = convert_to_cohere_tools(response_model)
        kwargs["tools"] = [tool_schema]

        return response_model, kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Extract and validate structured data from Cohere response."""
        # Extract logic from v1 process_response
        tool_calls = response.tool_calls or []
        if not tool_calls:
            raise ValueError("No tool calls in response")

        # Parse first tool call
        tool_call = tool_calls[0]
        return response_model.model_validate_json(
            tool_call.parameters,
            context=validation_context,
            strict=strict,
        )

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle validation failure and prepare retry."""
        # Extract logic from v1 reask_cohere_tools
        kwargs = kwargs.copy()
        error_msg = f"Validation Error: {exception}\nPlease fix and retry."
        kwargs["messages"].append({"role": "user", "content": error_msg})
        return kwargs

@register_mode_handler(Provider.COHERE, Mode.JSON)
class CohereJSONHandler(ModeHandler):
    """Handler for Cohere JSON mode."""
    # Similar structure for JSON mode
    ...
```

**Key Migration Patterns**:

1. **Request Preparation**: Move logic from `handle_*_modes()` functions
2. **Response Parsing**: Extract from `process_response()` or response handlers
3. **Reask Handling**: Move from `reask_*()` functions
4. **Error Handling**: Use Pydantic `ValidationError` for retries

#### Step 6: Create V2 Factory Function

Create the factory function using `patch_v2`:

```python
# instructor/v2/providers/your_provider/client.py
from instructor.v2.core.patch import patch_v2
from instructor import Instructor, AsyncInstructor, Mode, Provider
from instructor.v2.core.registry import mode_registry, normalize_mode
from typing import overload, Any

# Ensure handlers are registered (import triggers decorators)
from . import handlers  # noqa: F401

@overload
def from_cohere(
    client: cohere.Client,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...

@overload
def from_cohere(
    client: cohere.AsyncClient,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...

def from_cohere(
    client: cohere.Client | cohere.AsyncClient,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create v2 Instructor instance from Cohere client.

    Args:
        client: Cohere client instance (sync or async)
        mode: Mode to use (defaults to Mode.TOOLS)
        **kwargs: Additional kwargs for Instructor constructor

    Returns:
        Instructor instance (sync or async)
    """
    # Normalize provider-specific modes to generic modes
    normalized_mode = normalize_mode(Provider.COHERE, mode)

    # Validate mode is registered
    if not mode_registry.is_registered(Provider.COHERE, normalized_mode):
        from instructor.core.exceptions import ModeError
        available_modes = mode_registry.get_modes_for_provider(Provider.COHERE)
        raise ModeError(
            mode=mode.value,
            provider=Provider.COHERE.value,
            valid_modes=[m.value for m in available_modes],
        )

    # Determine sync/async
    is_async = isinstance(client, cohere.AsyncClient)

    # Get the API function to patch
    create_func = client.chat

    # Patch using v2 registry
    patched_create = patch_v2(
        func=create_func,
        provider=Provider.COHERE,
        mode=normalized_mode,
    )

    # Return appropriate instructor type
    if is_async:
        return AsyncInstructor(
            client=client,
            create=patched_create,
            provider=Provider.COHERE,
            mode=mode,  # Keep original mode for client
            **kwargs,
        )
    else:
        return Instructor(
            client=client,
            create=patched_create,
            provider=Provider.COHERE,
            mode=mode,
            **kwargs,
        )
```

**Key Differences from V1**:

- Uses `patch_v2()` instead of `instructor.patch()`
- Validates mode registration via registry
- Normalizes provider-specific modes
- Uses generic `Mode` enum values

#### Step 7: Export Provider

Update `__init__.py` to export the factory:

```python
# instructor/v2/providers/your_provider/__init__.py
from . import handlers  # noqa: F401 - triggers registration
from .client import from_cohere

__all__ = ["from_cohere"]
```

Update main v2 exports:

```python
# instructor/v2/__init__.py
try:
    from instructor.v2.providers.cohere import from_cohere
except ImportError:
    from_cohere = None  # type: ignore

__all__ = [
    # ... existing exports ...
    "from_cohere",
]
```

#### Step 8: Write Comprehensive Tests

Create tests following the testing guide (see "Testing Guide" section):

```python
# tests/v2/test_cohere_provider.py
import pytest
from pydantic import BaseModel
from instructor import Mode
from instructor.v2 import Provider, mode_registry

class TestModel(BaseModel):
    value: str

def test_mode_registration():
    """Verify modes are registered."""
    assert mode_registry.is_registered(Provider.COHERE, Mode.TOOLS)
    assert mode_registry.is_registered(Provider.COHERE, Mode.JSON)

@pytest.mark.requires_api_key
def test_basic_extraction():
    """Test end-to-end extraction."""
    from instructor.v2.providers.cohere import from_cohere
    import cohere

    client = cohere.Client(api_key="...")
    instructor_client = from_cohere(client, mode=Mode.TOOLS)

    result = instructor_client.create(
        response_model=TestModel,
        messages=[{"role": "user", "content": "Return value='test'"}],
    )

    assert isinstance(result, TestModel)
    assert result.value == "test"
```

#### Step 9: Update Integration Points

1. **Update `from_provider()` routing** (if applicable):
   - Ensure `instructor.from_provider("cohere/model")` routes to v2

2. **Add deprecation warnings** to v1 factory:

   ```python
   # instructor/providers/cohere/client.py
   def from_cohere(...):
       warnings.warn(
           "from_cohere() is deprecated. Use instructor.v2.providers.cohere.from_cohere()",
           DeprecationWarning,
           stacklevel=2,
       )
       # ... existing v1 code ...
   ```

3. **Update documentation**:
   - Add provider to migration checklist
   - Update examples to use v2

### Common Migration Patterns

#### Pattern 1: Simple Provider (No Custom Utils)

**V1**: Provider uses standard `instructor.patch()` with minimal customization.

**V2**: Create handlers that delegate to standard processing:

```python
@register_mode_handler(Provider.SIMPLE, Mode.TOOLS)
class SimpleToolsHandler(ModeHandler):
    def prepare_request(self, response_model, kwargs):
        # Minimal customization
        return response_model, kwargs

    def parse_response(self, response, response_model, **kwargs):
        # Use standard parsing
        return response_model.model_validate(response.data)

    def handle_reask(self, kwargs, response, exception):
        # Standard reask pattern
        kwargs["messages"].append({
            "role": "user",
            "content": f"Error: {exception}. Please fix."
        })
        return kwargs
```

#### Pattern 2: Provider with Complex Utils

**V1**: Provider has extensive utility functions in `utils.py`.

**V2**: Import and adapt existing utilities:

```python
from instructor.providers.cohere import utils as cohere_utils

@register_mode_handler(Provider.COHERE, Mode.JSON)
class CohereJSONHandler(ModeHandler):
    def prepare_request(self, response_model, kwargs):
        # Reuse v1 utility function
        return cohere_utils.handle_cohere_json_schema(response_model, kwargs)

    def handle_reask(self, kwargs, response, exception):
        # Reuse v1 reask function
        return cohere_utils.reask_cohere_tools(kwargs, response, exception)
```

#### Pattern 3: Provider with Multiple API Functions

**V1**: Provider patches different functions based on client type.

**V2**: Handle in factory function:

```python
def from_provider(client, mode=Mode.TOOLS):
    # Determine which function to patch
    if isinstance(client, SyncClient):
        create_func = client.chat
    elif isinstance(client, AsyncClient):
        create_func = client.chat_async
    else:
        raise ClientError("Invalid client type")

    patched_create = patch_v2(
        func=create_func,
        provider=Provider.YOUR_PROVIDER,
        mode=mode,
    )
    # ...
```

#### Pattern 4: Provider with Streaming Support

**V1**: Streaming handled in `process_response()`.

**V2**: Check for streaming in handler:

```python
class ProviderToolsHandler(ModeHandler):
    def prepare_request(self, response_model, kwargs):
        # Register streaming model if stream=True
        if kwargs.get("stream") and response_model:
            self._streaming_models[response_model] = None
        return response_model, kwargs

    def parse_response(self, response, response_model, **kwargs):
        # Check if this is a streaming response
        if response_model in self._streaming_models:
            return response_model.from_streaming_response(response)
        # Normal parsing
        return response_model.model_validate(...)
```

### Migration Checklist

Use this checklist when migrating a provider:

**Pre-Migration**:

- [ ] Understand v1 implementation structure
- [ ] Identify all supported modes
- [ ] Map provider-specific modes to generic modes
- [ ] Identify request preparation logic
- [ ] Identify response parsing logic
- [ ] Identify reask/retry logic

**Implementation**:

- [ ] Create v2 provider directory structure
- [ ] Implement handler classes with `@register_mode_handler`
- [ ] Implement `prepare_request()` method
- [ ] Implement `parse_response()` method
- [ ] Implement `handle_reask()` method
- [ ] Create factory function using `patch_v2()`
- [ ] Add proper type hints and overloads
- [ ] Export provider in `__init__.py`

**Testing**:

- [ ] Test mode registration
- [ ] Test basic extraction (sync)
- [ ] Test basic extraction (async)
- [ ] Test all supported modes
- [ ] Test error handling
- [ ] Test retry logic
- [ ] Test streaming (if applicable)
- [ ] Test edge cases

**Integration**:

- [ ] Update `from_provider()` routing (if needed)
- [ ] Add deprecation warnings to v1 factory
- [ ] Update migration checklist in README
- [ ] Update documentation
- [ ] Verify backward compatibility

**Post-Migration**:

- [ ] Monitor for issues
- [ ] Collect user feedback
- [ ] Plan v1 deprecation timeline

### Troubleshooting Common Issues

**Issue**: Mode not found in registry

- **Solution**: Ensure handlers module is imported before using factory (use `# noqa: F401` import)

**Issue**: Handler methods not being called

- **Solution**: Verify `@register_mode_handler` decorator is applied correctly and module is imported

**Issue**: Provider-specific modes not working

- **Solution**: Ensure `normalize_mode()` handles your provider-specific modes

**Issue**: Tests failing with import errors

- **Solution**: Ensure provider handlers are imported in test files or use `from . import handlers`

**Issue**: Async client not working

- **Solution**: Verify `is_async()` check and use `AsyncInstructor` for async clients

### Migration Example: Complete Cohere Migration

See `instructor/v2/providers/anthropic/` and `instructor/v2/providers/genai/` for complete reference implementations.

### Key Differences Summary

| Aspect                   | V1                                    | V2                                   |
| ------------------------ | ------------------------------------- | ------------------------------------ |
| **Mode Handling**        | Direct calls in `process_response()`  | Registry-based handler lookup        |
| **Mode Validation**      | Runtime (in factory function)         | Compile-time (in `patch_v2`)         |
| **Handler Organization** | Scattered utility functions           | Centralized handler classes          |
| **Mode Enums**           | Provider-specific (`ANTHROPIC_TOOLS`) | Generic (`TOOLS`) with normalization |
| **Registration**         | Manual function calls                 | Decorator-based auto-registration    |
| **Testing**              | Test entire flow                      | Test handlers independently          |

V1 code continues to work during transition period, but new code should use v2.

## How the System Works

### Request Flow

When a user calls `client.create(response_model=MyModel, ...)`, the following happens:

1. **Patch Time** (`patch_v2`):
   - Validates that the mode is registered for the provider
   - Creates a wrapper function that intercepts calls
   - Injects default model if provided

2. **Request Preparation** (`prepare_request`):
   - Handler receives `response_model` and request `kwargs`
   - Converts `response_model` to provider-specific format (e.g., tools schema for TOOLS mode)
   - Modifies `kwargs` to include provider-specific parameters
   - Returns modified `response_model` and `kwargs`

3. **API Call**:
   - Original provider API function is called with modified kwargs
   - Returns raw provider response object

4. **Response Parsing** (`parse_response`):
   - Handler extracts structured data from raw response
   - Validates against `response_model` using Pydantic
   - Returns validated Pydantic model instance

5. **Retry on Failure** (`handle_reask`):
   - If validation fails, handler modifies kwargs with error context
   - Retry logic calls API again with updated kwargs
   - Process repeats up to `max_retries` times

### Mode Normalization

Provider-specific modes (e.g., `Mode.ANTHROPIC_TOOLS`) are automatically normalized to generic modes (e.g., `Mode.TOOLS`) for registry lookup. This allows:

- Backward compatibility with provider-specific mode names
- Shared handler implementations across providers
- Consistent mode semantics

The normalization happens in `normalize_mode()` function in `registry.py`.

### Handler Lifecycle

1. **Registration**: Handler classes decorated with `@register_mode_handler` are instantiated and registered when the module is imported
2. **Lookup**: When a request is made, handlers are retrieved from the registry using `(Provider, Mode)` tuple
3. **Execution**: Handler methods are called during request preparation, response parsing, and retry handling
4. **Caching**: Handlers are cached in the registry after first lookup for performance

### Registry Internals

The registry stores handlers in a dictionary keyed by `(Provider, Mode)` tuples:

```python
{
    (Provider.ANTHROPIC, Mode.TOOLS): ModeHandlers(...),
    (Provider.ANTHROPIC, Mode.JSON): ModeHandlers(...),
    (Provider.GENAI, Mode.TOOLS): ModeHandlers(...),
    ...
}
```

Each `ModeHandlers` object contains:

- `request_handler`: Function to prepare request kwargs
- `reask_handler`: Function to handle validation failures
- `response_parser`: Function to parse API responses

## Testing Guide

### Writing Tests for V2 Providers

Tests for v2 providers should verify:

1. Mode registration in the registry
2. Handler functionality (request preparation, response parsing, reask handling)
3. End-to-end extraction with real API calls
4. Error handling and retry logic

### Test Structure

Create tests in `tests/v2/` directory following this pattern:

```python
"""Tests for YourProvider v2 implementation."""

import pytest
from pydantic import BaseModel
from instructor import Mode
from instructor.v2 import Provider, mode_registry

class SimpleModel(BaseModel):
    """Simple test model."""
    value: str

# Test mode registration
def test_mode_is_registered():
    """Verify mode is registered in the v2 registry."""
    assert mode_registry.is_registered(Provider.YOUR_PROVIDER, Mode.TOOLS)

    handlers = mode_registry.get_handlers(Provider.YOUR_PROVIDER, Mode.TOOLS)
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None

# Test basic extraction
@pytest.mark.requires_api_key
def test_basic_extraction():
    """Test basic extraction with real API call."""
    from instructor.v2.providers.your_provider import from_your_provider
    from your_provider_sdk import Client

    client = Client(api_key="...")
    instructor_client = from_your_provider(client, mode=Mode.TOOLS)

    result = instructor_client.create(
        response_model=SimpleModel,
        messages=[{"role": "user", "content": "Return value='test'"}],
    )

    assert isinstance(result, SimpleModel)
    assert result.value == "test"

# Test async extraction
@pytest.mark.asyncio
@pytest.mark.requires_api_key
async def test_async_extraction():
    """Test async extraction."""
    from instructor.v2.providers.your_provider import from_your_provider
    from your_provider_sdk import AsyncClient

    client = AsyncClient(api_key="...")
    instructor_client = from_your_provider(client, mode=Mode.TOOLS)

    result = await instructor_client.create(
        response_model=SimpleModel,
        messages=[{"role": "user", "content": "Return value='async'"}],
    )

    assert isinstance(result, SimpleModel)
    assert result.value == "async"
```

### Parametrized Tests

Use pytest parametrization to test multiple modes:

```python
@pytest.mark.parametrize(
    "provider,mode",
    [
        (Provider.YOUR_PROVIDER, Mode.TOOLS),
        (Provider.YOUR_PROVIDER, Mode.JSON),
    ],
)
@pytest.mark.requires_api_key
def test_all_modes(provider: Provider, mode: Mode):
    """Test all registered modes."""
    # Test implementation
    pass
```

### Testing Handler Methods Directly

You can test handler methods in isolation:

```python
def test_handler_prepare_request():
    """Test request preparation logic."""
    from instructor.v2.providers.your_provider.handlers import YourProviderToolsHandler

    handler = YourProviderToolsHandler()
    response_model, kwargs = handler.prepare_request(
        response_model=SimpleModel,
        kwargs={"messages": [{"role": "user", "content": "test"}]},
    )

    assert "tools" in kwargs  # Verify tools were added
    assert response_model == SimpleModel

def test_handler_parse_response():
    """Test response parsing logic."""
    from instructor.v2.providers.your_provider.handlers import YourProviderToolsHandler

    handler = YourProviderToolsHandler()
    # Mock response object
    mock_response = create_mock_response(...)

    result = handler.parse_response(
        response=mock_response,
        response_model=SimpleModel,
    )

    assert isinstance(result, SimpleModel)
```

### Test Coverage Checklist

For each provider mode, ensure tests cover:

- [ ] Mode registration verification
- [ ] Basic extraction (sync)
- [ ] Basic extraction (async)
- [ ] Request preparation (handler method)
- [ ] Response parsing (handler method)
- [ ] Reask handling (handler method)
- [ ] Error handling (invalid responses)
- [ ] Retry logic (validation failures)
- [ ] Streaming support (if applicable)
- [ ] Mode-specific features (e.g., parallel tools, thinking)

### Running Tests

```bash
# Run all v2 tests
pytest tests/v2/ -v

# Run tests for specific provider
pytest tests/v2/test_provider_modes.py -v

# Run with API key (requires environment variable)
ANTHROPIC_API_KEY=... pytest tests/v2/ -v -m requires_api_key
```

## Provider Migration Checklist

This checklist tracks which providers have been migrated to v2:

### Completed Migrations

- [x] **Anthropic** (`Provider.ANTHROPIC`)
  - Location: `instructor/v2/providers/anthropic/`
  - Modes: `TOOLS`, `JSON_SCHEMA`, `PARALLEL_TOOLS`, `ANTHROPIC_REASONING_TOOLS` (deprecated)
  - Tests: `tests/v2/test_provider_modes.py`
  - Status: ✅ Complete

- [x] **Google GenAI** (`Provider.GENAI`)
  - Location: `instructor/v2/providers/genai/`
  - Modes: `TOOLS`, `JSON`
  - Tests: `tests/v2/test_provider_modes.py`, `tests/v2/test_genai_integration.py`
  - Status: ✅ Complete

- [x] **OpenAI** (`Provider.OPENAI`)
  - Location: `instructor/v2/providers/openai/`
  - Modes: `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS`, `RESPONSES_TOOLS`
  - Tests: `tests/v2/test_provider_modes.py`, `tests/v2/test_openai_streaming.py`
  - Status: ✅ Complete

- [x] **Cohere** (`Provider.COHERE`)
  - Location: `instructor/v2/providers/cohere/`
  - Modes: `TOOLS`, `JSON_SCHEMA`, `MD_JSON`
  - Tests: `tests/v2/test_provider_modes.py`, `tests/v2/test_cohere_handlers.py`
  - Status: ✅ Complete

- [x] **Mistral** (`Provider.MISTRAL`)
  - Location: `instructor/v2/providers/mistral/`
  - Modes: `TOOLS`, `JSON_SCHEMA`, `MD_JSON`
  - Tests: `tests/v2/test_provider_modes.py`, `tests/v2/test_mistral_client.py`, `tests/v2/test_mistral_handlers.py`
  - Status: ✅ Complete

- [x] **Groq** (`Provider.GROQ`)
  - Location: `instructor/v2/providers/groq/`
  - Modes: `TOOLS`, `MD_JSON`
  - Tests: `tests/v2/test_provider_modes.py`, `tests/v2/test_groq_client.py`, `tests/v2/test_groq_handlers.py`
  - Status: ✅ Complete

- [x] **Fireworks** (`Provider.FIREWORKS`)
  - Location: `instructor/v2/providers/fireworks/`
  - Modes: `TOOLS`, `MD_JSON`
  - Tests: `tests/v2/test_provider_modes.py`, `tests/v2/test_fireworks_client.py`, `tests/v2/test_fireworks_handlers.py`
  - Status: ✅ Complete

- [x] **Cerebras** (`Provider.CEREBRAS`)
  - Location: `instructor/v2/providers/cerebras/`
  - Modes: `TOOLS`, `MD_JSON`
  - Tests: `tests/v2/test_provider_modes.py`, `tests/v2/test_cerebras_client.py`, `tests/v2/test_cerebras_handlers.py`
  - Status: ✅ Complete

- [x] **Writer** (`Provider.WRITER`)
  - Location: `instructor/v2/providers/writer/`
  - Modes: `TOOLS`, `MD_JSON`
  - Tests: `tests/v2/test_provider_modes.py`, `tests/v2/test_writer_client.py`, `tests/v2/test_writer_handlers.py`
  - Status: ✅ Complete

- [x] **xAI** (`Provider.XAI`)
  - Location: `instructor/v2/providers/xai/`
  - Modes: `TOOLS`, `JSON_SCHEMA`, `MD_JSON`
  - Tests: `tests/v2/test_provider_modes.py`, `tests/v2/test_xai_client.py`, `tests/v2/test_xai_handlers.py`
  - Status: ✅ Complete

- [x] **Bedrock** (`Provider.BEDROCK`)
  - Location: `instructor/v2/providers/bedrock/`
  - Modes: `TOOLS`, `MD_JSON`
  - Tests: `tests/v2/test_provider_modes.py`, `tests/v2/test_bedrock_client.py`, `tests/v2/test_bedrock_handlers.py`
  - Status: ✅ Complete

### Pending Migrations

The following providers exist in v1 (`instructor/providers/`) but have not yet been migrated to v2:

- [ ] **Google Gemini** (`Provider.GEMINI`)
  - Location: `instructor/providers/gemini/`
  - Modes: TBD
  - Priority: Medium

- [ ] **Perplexity** (`Provider.PERPLEXITY`)
  - Location: `instructor/providers/perplexity/`
  - Modes: TBD
  - Priority: Low

- [ ] **Vertex AI** (`Provider.VERTEXAI`)
  - Location: `instructor/providers/vertexai/`
  - Modes: TBD
  - Priority: Medium

### Migration Steps

To migrate a provider to v2:

1. **Create provider directory**: `instructor/v2/providers/your_provider/`
2. **Implement handlers**: Create `handlers.py` with `@register_mode_handler` decorators
3. **Create factory function**: Create `client.py` with `from_your_provider()` function
4. **Export**: Update `__init__.py` to export the factory function
5. **Add to v2 exports**: Update `instructor/v2/__init__.py` to import provider
6. **Write tests**: Create tests in `tests/v2/` following the testing guide above
7. **Update checklist**: Mark provider as complete in this document

### Migration Notes

- Providers can coexist in v1 and v2 during migration
- Use `instructor.from_provider()` which routes to v2 when available
- Test both sync and async clients
- Verify all modes work correctly
- Ensure backward compatibility with existing code

## Best Practices

- **New Modes**: Define in `instructor.Mode` enum, create handler, register via decorator
- **Error Handling**: Validate early, provide context, preserve exception chains
- **Testing**: Test both success and failure paths, verify registry registration
- **Documentation**: Document provider-specific behavior in handler docstrings
- **Type Safety**: Use type hints throughout handler implementations

## Module Organization

```text
instructor/v2/
├── __init__.py              # V2 exports (ModeHandler, Protocols, Registry, Providers)
├── README.md               # This document
├── core/
│   ├── __init__.py         # Core exports (Protocols, Registry)
│   ├── decorators.py       # @register_mode_handler decorator
│   ├── exceptions.py       # Exception classes & validation utilities
│   ├── handler.py          # ModeHandler abstract base class
│   ├── patch.py           # Patching mechanism
│   ├── protocols.py       # Protocol definitions (RequestHandler, etc.)
│   ├── registry.py        # Mode registry implementation
│   └── retry.py           # Retry logic (sync & async)
└── providers/
    ├── __init__.py         # Provider exports
    └── anthropic/          # Anthropic provider implementation
        ├── __init__.py     # Provider exports
        ├── client.py       # from_anthropic factory function
        └── handlers.py     # Handler implementations (TOOLS, JSON, etc.)
```

## Module Exports

- `instructor.v2`: `ModeHandler`, `mode_registry`, `RequestHandler`, `ReaskHandler`, `ResponseParser`, `from_anthropic`
- `instructor.v2.core`: Core types and registry
- `instructor.v2.providers.anthropic`: `from_anthropic`
