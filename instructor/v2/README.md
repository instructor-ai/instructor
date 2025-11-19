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

## Migration from V1 to V2

**Key Differences**:

- Handler dispatch: Registry lookup (v2) vs direct calls (v1)
- Mode validation: At patch time (v2) vs runtime (v1)
- Exception context: Enhanced with attempt tracking (v2)

**Migration**: Add explicit `mode` parameter to factory functions:

```python
# V1
client = from_anthropic(anthropic_client)

# V2
client = from_anthropic(anthropic_client, mode=Mode.TOOLS)
```

V1 code continues to work during transition period.

## Best Practices

- **New Modes**: Define in `instructor.Mode` enum, create handler, register via decorator
- **Error Handling**: Validate early, provide context, preserve exception chains
- **Testing**: Test both success and failure paths, verify registry registration

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