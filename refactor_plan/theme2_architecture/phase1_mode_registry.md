# Phase 1: Mode Handler Registry

**Status**: Not Started
**Priority**: P0 (Foundation for Theme 2)
**Est. Duration**: 4-6 weeks
**Est. Effort**: 20-25 days
**Assignee**: TBD
**Dependencies**: None (can start immediately, but Theme 1 Phase 2 helps)

---

## Quick Reference

| Component | Current | Target | Impact |
|-----------|---------|--------|--------|
| Dispatch dicts | 3 separate | 1 registry | Single source of truth |
| Mode handlers | 37 hardcoded | Dynamic registration | Extensible |
| Provider imports | All at load (11 files) | Lazy on demand | 90% faster imports |
| Type safety | Defeated (`# type: ignore`) | Protocol-based | Compile-time checks |
| Lines of code | 74 entries across 2 dicts | ~150 lines registry | Cleaner |

---

## Overview

Replace hardcoded mode dispatch dictionaries with a dynamic registry system that enables lazy loading, provider independence, and type safety.

### Current Problems

**File**: `instructor/processing/response.py`

**Problem 1: Three Separate Dispatch Dictionaries** (Lines 405-663)

```python
# 1. PARALLEL_MODES (lines 405-409)
PARALLEL_MODES = {
    Mode.PARALLEL_TOOLS: handle_parallel_tools,
    Mode.VERTEXAI_PARALLEL_TOOLS: handle_vertexai_parallel_tools,
    Mode.ANTHROPIC_PARALLEL_TOOLS: handle_anthropic_parallel_tools,
}

# 2. mode_handlers (lines 432-469)
mode_handlers = {  # type: ignore
    Mode.FUNCTIONS: handle_functions,
    Mode.TOOLS_STRICT: handle_tools_strict,
    # ... 32 more entries
}

# 3. REASK_HANDLERS (lines 612-663)
REASK_HANDLERS = {
    Mode.FUNCTIONS: reask_default,
    Mode.TOOLS_STRICT: reask_tools,
    # ... 35 more entries
}
```

**Issues**:
- Must manually keep 3 dictionaries in sync
- Adding a mode requires editing response.py in 3 places
- No validation that all modes have all handlers
- Dictionary recreated on every request (mode_handlers)

**Problem 2: All Provider Utils Loaded Eagerly** (Lines 59-161)

```python
from ..providers.anthropic.utils import (
    handle_anthropic_json,
    handle_anthropic_parallel_tools,
    handle_anthropic_reasoning_tools,
    handle_anthropic_tools,
    reask_anthropic_json,
    reask_anthropic_tools,
)
# ... repeated for 10 more providers (103 lines of imports)
```

**Issues**:
- All 11 provider utils files loaded on `import instructor`
- 3,488 lines of code loaded even if only using OpenAI
- ~500ms import time

**Problem 3: Type Safety Defeated** (Line 432)

```python
mode_handlers = {  # type: ignore
    Mode.FUNCTIONS: handle_functions,
    # ... can't verify completeness
}
```

**Issues**:
- Type checker can't verify all modes have handlers
- Missing mode only discovered at runtime
- No compile-time safety

### Goals

1. **Hierarchical Design**: Separate Provider and ModeType enums instead of flat Mode enum
2. **Single Registry**: One source of truth for mode handlers
3. **Lazy Loading**: Load handlers only when mode is used
4. **Type Safety**: Protocol-based validation at registration
5. **Provider Independence**: Providers register their own modes
6. **Composability**: Any (Provider, ModeType) combination vs 42 hardcoded modes
7. **Backward Compatible**: Support old code during migration

### Success Metrics

- ☐ Single ModeRegistry replaces 3 dispatch dictionaries
- ☐ All 37 modes migrated to registry
- ☐ Import time <100ms (with lazy loading)
- ☐ Type checker passes without `# type: ignore`
- ☐ All tests passing
- ☐ Old code path still works (backward compatibility)

---

## Current State Analysis

### Mode Handler Locations

**From MEASUREMENTS.md**:

| Dictionary | Lines | Modes | Purpose |
|------------|-------|-------|---------|
| PARALLEL_MODES | 405-409 | 3 | Parallel tool calling |
| mode_handlers | 432-469 | 34 | Request preparation |
| REASK_HANDLERS | 612-663 | 37 | Error recovery |

### Mode Distribution by Provider

| Provider | Modes | Examples |
|----------|-------|----------|
| OpenAI | 10 | TOOLS, JSON, PARALLEL_TOOLS, RESPONSES_TOOLS |
| Anthropic | 4 | ANTHROPIC_TOOLS, ANTHROPIC_JSON, ANTHROPIC_PARALLEL_TOOLS |
| Google/Vertex | 7 | GEMINI_JSON, GEMINI_TOOLS, VERTEXAI_TOOLS |
| Mistral | 2 | MISTRAL_TOOLS, MISTRAL_STRUCTURED_OUTPUTS |
| Cohere | 2 | COHERE_TOOLS, COHERE_JSON_SCHEMA |
| Others | 12 | Various |

### Handler Signature Analysis

**Request Handler** (from mode_handlers):
```python
def handle_*_tools(
    response_model: type[BaseModel],
    kwargs: dict[str, Any]
) -> tuple[type[BaseModel], dict[str, Any]]:
    """Prepare request for this mode."""
    # Modify kwargs (add tools, response_format, etc.)
    return response_model, kwargs
```

**Reask Handler** (from REASK_HANDLERS):
```python
def reask_*(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception
) -> dict[str, Any]:
    """Handle validation failure."""
    # Append error message to kwargs
    return kwargs
```

**Response Processor** (currently mixed into handlers):
```python
def process_response_*(
    response: Any,
    response_model: type[BaseModel],
    **kwargs
) -> BaseModel:
    """Extract and validate model from response."""
    # Parse response based on mode
    return validated_model
```

---

## Hierarchical Mode Design (NEW)

### Flat vs Hierarchical

**Current** (Flat - 42 separate enums):
```python
class Mode(Enum):
    GEMINI_JSON = "gemini_json"
    GEMINI_TOOLS = "gemini_tools"
    OPENAI_JSON = "json_mode"
    OPENAI_TOOLS = "tool_call"
    ANTHROPIC_JSON = "anthropic_json"
    ANTHROPIC_TOOLS = "anthropic_tools"
    # ... 36 more modes
```

**Issues with flat design**:
- 42 separate mode enums
- Adding provider requires N new mode enums
- Can't query "all JSON modes" or "all Anthropic modes"
- Mode names inconsistent (`json_mode` vs `gemini_json`)
- Not composable

**New** (Hierarchical - Provider + ModeType):
```python
class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    COHERE = "cohere"
    # ... ~15 providers

class ModeType(Enum):
    TOOLS = "tools"
    JSON = "json"
    PARALLEL_TOOLS = "parallel"
    STRUCTURED_OUTPUTS = "structured"
    REASONING_TOOLS = "reasoning"
    # ... ~6 mode types

# Composite mode = (Provider, ModeType)
type Mode = tuple[Provider, ModeType]

# Usage
mode = (Provider.GEMINI, ModeType.JSON)
```

**Benefits**:
1. **Fewer enums**: 15 + 6 = 21 enums vs 42
2. **Composable**: Any (Provider, ModeType) combination
3. **Queryable**: "All JSON modes" = `filter by ModeType.JSON`
4. **Clear semantics**: `(GEMINI, JSON)` is clearer than `GEMINI_JSON`
5. **Easier to add providers**: Just register existing mode types
6. **Provider-agnostic code**: Can use `ModeType.TOOLS` across providers

### Registry with Composite Keys

Registry uses `(Provider, ModeType)` as keys instead of flat `Mode`:

```python
# Registration
mode_registry.register(
    provider=Provider.GEMINI,
    mode_type=ModeType.JSON,
    handler=GeminiJSONHandler()
)

# Lookup
handler = mode_registry.get_handler(Provider.GEMINI, ModeType.JSON)

# Query by provider
gemini_modes = mode_registry.get_modes_for_provider(Provider.GEMINI)
# → [ModeType.JSON, ModeType.TOOLS]

# Query by mode type
json_providers = mode_registry.get_providers_for_mode(ModeType.JSON)
# → [Provider.GEMINI, Provider.OPENAI, Provider.ANTHROPIC, ...]
```

---

## Proposed Solution

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ModeRegistry                              │
│  - _handlers: dict[tuple[Provider, ModeType], ModeHandler]      │
│  - _lazy_loaders: dict[tuple[Provider, ModeType], Callable]     │
│                                                                  │
│  + register(provider, mode_type, handler?, lazy_loader?)        │
│  + get_handler(provider, mode_type) -> ModeHandler              │
│  + get_modes_for_provider(provider) -> list[ModeType]           │
│  + get_providers_for_mode(mode_type) -> list[Provider]          │
│  + has_mode(provider, mode_type) -> bool                        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ uses
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   ModeHandler Protocol                   │
│  + prepare_request(response_model, kwargs)              │
│  + process_response(response, response_model, **kwargs) │
│  + handle_reask(kwargs, response, exception)            │
│  + supports_streaming: bool                             │
│  + supports_parallel: bool                              │
└─────────────────────────────────────────────────────────┘
                            │
                            │ implements
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Provider-Specific Handlers                  │
│  AnthropicToolsHandler, OpenAIToolsHandler, etc.        │
└─────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. ModeHandler Protocol

**File**: `instructor/core/mode_handler.py` (new)

```python
from typing import Protocol, Any, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class ModeHandler(Protocol):
    """
    Protocol defining the interface for mode handlers.

    All mode handlers must implement these methods to be registered
    with the ModeRegistry.
    """

    def prepare_request(
        self,
        response_model: type[T],
        kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """
        Prepare the API request for this mode.

        Modifies kwargs to add mode-specific parameters like tools,
        response_format, etc.

        Args:
            response_model: The Pydantic model to extract
            kwargs: The API call kwargs

        Returns:
            Tuple of (possibly modified response_model, modified kwargs)
        """
        ...

    def process_response(
        self,
        response: Any,
        response_model: type[T],
        **kwargs: Any
    ) -> T:
        """
        Process the API response to extract the model.

        Args:
            response: The raw API response
            response_model: The Pydantic model to extract
            **kwargs: Additional context

        Returns:
            Validated instance of response_model
        """
        ...

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception
    ) -> dict[str, Any]:
        """
        Handle validation failure by modifying kwargs for retry.

        Args:
            kwargs: The original API call kwargs
            response: The failed response
            exception: The validation exception

        Returns:
            Modified kwargs for retry with error message
        """
        ...

    @property
    def supports_streaming(self) -> bool:
        """Whether this mode supports streaming responses."""
        return False

    @property
    def supports_parallel(self) -> bool:
        """Whether this mode supports parallel tool calls."""
        return False

    @property
    def provider(self) -> str:
        """Provider this mode belongs to (e.g., 'openai', 'anthropic')."""
        return "unknown"
```

#### 2. ModeRegistry

**File**: `instructor/core/mode_registry.py` (new)

```python
from typing import Callable, Protocol
from instructor.mode import Mode
from instructor.core.mode_handler import ModeHandler
import logging

logger = logging.getLogger(__name__)

class ModeRegistry:
    """
    Central registry for mode handlers with lazy loading support.

    Example:
        # Register eagerly
        registry.register(Mode.OPENAI_TOOLS, OpenAIToolsHandler())

        # Register lazily (handler loaded on first use)
        registry.register(
            Mode.ANTHROPIC_TOOLS,
            lazy_loader=lambda: AnthropicToolsHandler()
        )

        # Get handler
        handler = registry.get_handler(Mode.OPENAI_TOOLS)
    """

    def __init__(self):
        self._handlers: dict[Mode, ModeHandler] = {}
        self._lazy_loaders: dict[Mode, Callable[[], ModeHandler]] = {}
        self._loading: set[Mode] = set()  # Prevent circular loading

    def register(
        self,
        mode: Mode,
        handler: ModeHandler | None = None,
        lazy_loader: Callable[[], ModeHandler] | None = None,
        replace: bool = False
    ) -> None:
        """
        Register a handler for a mode.

        Args:
            mode: The mode to register
            handler: Pre-instantiated handler (loaded immediately)
            lazy_loader: Function to create handler (loaded on first use)
            replace: If True, replace existing handler. If False, raise error.

        Raises:
            ValueError: If neither handler nor lazy_loader provided
            ValueError: If mode already registered and replace=False
        """
        if handler is None and lazy_loader is None:
            raise ValueError(
                f"Must provide either handler or lazy_loader for mode {mode}"
            )

        # Check if already registered
        if mode in self._handlers or mode in self._lazy_loaders:
            if not replace:
                raise ValueError(
                    f"Mode {mode} is already registered. "
                    f"Use replace=True to override."
                )
            logger.warning(f"Replacing existing handler for mode {mode}")

        if handler is not None:
            # Validate handler implements protocol
            if not isinstance(handler, ModeHandler):
                # Try to check methods
                for method in ['prepare_request', 'process_response', 'handle_reask']:
                    if not hasattr(handler, method):
                        raise TypeError(
                            f"Handler for {mode} missing required method: {method}"
                        )

            self._handlers[mode] = handler
            # Remove lazy loader if exists
            self._lazy_loaders.pop(mode, None)
            logger.debug(f"Registered handler for mode {mode}")
        else:
            self._lazy_loaders[mode] = lazy_loader
            logger.debug(f"Registered lazy loader for mode {mode}")

    def get_handler(self, mode: Mode) -> ModeHandler:
        """
        Get handler for mode, loading lazily if needed.

        Args:
            mode: The mode to get handler for

        Returns:
            The mode handler

        Raises:
            ValueError: If no handler registered for mode
            RuntimeError: If circular loading detected
        """
        # Check if already loaded
        if mode in self._handlers:
            return self._handlers[mode]

        # Check if lazy loader exists
        if mode in self._lazy_loaders:
            # Detect circular loading
            if mode in self._loading:
                raise RuntimeError(
                    f"Circular loading detected for mode {mode}"
                )

            try:
                self._loading.add(mode)
                logger.debug(f"Lazy loading handler for mode {mode}")

                # Load the handler
                handler = self._lazy_loaders[mode]()

                # Validate
                if not isinstance(handler, ModeHandler):
                    for method in ['prepare_request', 'process_response', 'handle_reask']:
                        if not hasattr(handler, method):
                            raise TypeError(
                                f"Lazy-loaded handler for {mode} missing: {method}"
                            )

                # Store and remove lazy loader
                self._handlers[mode] = handler
                del self._lazy_loaders[mode]

                logger.debug(f"Successfully loaded handler for mode {mode}")
                return handler
            finally:
                self._loading.discard(mode)

        # Not registered
        available = self.list_modes()
        raise ValueError(
            f"No handler registered for mode {mode}. "
            f"Available modes: {', '.join(str(m) for m in available[:5])}..."
            if len(available) > 5
            else f"Available modes: {', '.join(str(m) for m in available)}"
        )

    def has_mode(self, mode: Mode) -> bool:
        """Check if a mode is registered."""
        return mode in self._handlers or mode in self._lazy_loaders

    def list_modes(self) -> list[Mode]:
        """List all registered modes (both loaded and lazy)."""
        return sorted(
            list(self._handlers.keys()) + list(self._lazy_loaders.keys()),
            key=lambda m: m.value
        )

    def unregister(self, mode: Mode) -> None:
        """
        Unregister a mode (for testing or dynamic updates).

        Args:
            mode: The mode to unregister
        """
        self._handlers.pop(mode, None)
        self._lazy_loaders.pop(mode, None)
        logger.debug(f"Unregistered mode {mode}")

    def stats(self) -> dict[str, int]:
        """Get registry statistics."""
        return {
            "loaded": len(self._handlers),
            "lazy": len(self._lazy_loaders),
            "total": len(self._handlers) + len(self._lazy_loaders)
        }

# Global registry instance
mode_registry = ModeRegistry()

# Decorator for easy registration
def register_mode_handler(
    mode: Mode,
    lazy: bool = True,
    replace: bool = False
):
    """
    Decorator to register a mode handler class.

    Args:
        mode: The mode this handler is for
        lazy: If True, create instance on first use. If False, create immediately.
        replace: If True, replace existing handler

    Example:
        @register_mode_handler(Mode.OPENAI_TOOLS, lazy=False)
        class OpenAIToolsHandler:
            def prepare_request(self, response_model, kwargs):
                ...
    """
    def decorator(handler_class):
        if lazy:
            mode_registry.register(
                mode,
                lazy_loader=lambda: handler_class(),
                replace=replace
            )
        else:
            mode_registry.register(
                mode,
                handler=handler_class(),
                replace=replace
            )
        return handler_class
    return decorator
```

#### 3. Base Mode Handler Implementation

**File**: `instructor/core/base_mode_handler.py` (new)

```python
from typing import Any, TypeVar
from pydantic import BaseModel
from instructor.core.mode_handler import ModeHandler

T = TypeVar('T', bound=BaseModel)

class BaseModeHandler:
    """
    Base implementation of ModeHandler with common functionality.

    Providers can inherit from this to reduce boilerplate.
    """

    def __init__(self, provider: str):
        self._provider = provider

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_parallel(self) -> bool:
        return False

    def prepare_request(
        self,
        response_model: type[T],
        kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Default: no modification."""
        return response_model, kwargs

    def process_response(
        self,
        response: Any,
        response_model: type[T],
        **kwargs: Any
    ) -> T:
        """Default: assume response is already the model."""
        if isinstance(response, response_model):
            return response
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement process_response"
        )

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception
    ) -> dict[str, Any]:
        """Default: append error message to messages."""
        kwargs_copy = kwargs.copy()
        messages = kwargs_copy.get("messages", [])
        messages.append({
            "role": "user",
            "content": f"Validation failed: {exception}. Please fix and retry."
        })
        kwargs_copy["messages"] = messages
        return kwargs_copy
```

---

## Implementation Steps

### Step 1: Create Core Infrastructure (Week 1)

**Create 3 new files**:

1. `instructor/core/mode_handler.py` - Protocol definition
2. `instructor/core/mode_registry.py` - Registry implementation
3. `instructor/core/base_mode_handler.py` - Base class

**Tasks**:
- [ ] Copy code from "Proposed Solution" section above
- [ ] Add comprehensive docstrings
- [ ] Add type annotations
- [ ] Run `uv run ty check` to verify types

**Testing**:
```python
# tests/core/test_mode_registry.py
import pytest
from instructor.mode import Mode
from instructor.core.mode_registry import ModeRegistry, register_mode_handler
from instructor.core.mode_handler import ModeHandler

def test_register_eager():
    registry = ModeRegistry()

    class TestHandler:
        def prepare_request(self, response_model, kwargs):
            return response_model, kwargs
        def process_response(self, response, response_model, **kwargs):
            return response
        def handle_reask(self, kwargs, response, exception):
            return kwargs

    handler = TestHandler()
    registry.register(Mode.TOOLS, handler=handler)

    assert registry.has_mode(Mode.TOOLS)
    assert registry.get_handler(Mode.TOOLS) is handler

def test_register_lazy():
    registry = ModeRegistry()

    loaded = []

    def create_handler():
        loaded.append(True)
        class TestHandler:
            def prepare_request(self, response_model, kwargs):
                return response_model, kwargs
            def process_response(self, response, response_model, **kwargs):
                return response
            def handle_reask(self, kwargs, response, exception):
                return kwargs
        return TestHandler()

    registry.register(Mode.TOOLS, lazy_loader=create_handler)

    assert registry.has_mode(Mode.TOOLS)
    assert len(loaded) == 0  # Not loaded yet

    handler = registry.get_handler(Mode.TOOLS)
    assert len(loaded) == 1  # Now loaded

    # Second get should reuse
    handler2 = registry.get_handler(Mode.TOOLS)
    assert handler is handler2
    assert len(loaded) == 1  # Still only loaded once

def test_missing_mode_error():
    registry = ModeRegistry()

    with pytest.raises(ValueError, match="No handler registered"):
        registry.get_handler(Mode.TOOLS)

def test_decorator():
    registry = ModeRegistry()

    @register_mode_handler(Mode.TOOLS, lazy=False)
    class TestHandler:
        def prepare_request(self, response_model, kwargs):
            return response_model, kwargs
        def process_response(self, response, response_model, **kwargs):
            return response
        def handle_reask(self, kwargs, response, exception):
            return kwargs

    assert registry.has_mode(Mode.TOOLS)
```

Run tests:
```bash
uv run pytest tests/core/test_mode_registry.py -v
```

---

### Step 2: Migrate One Provider (Week 2)

**Choose OpenAI** (simplest, most common)

**Current**: `instructor/providers/openai/utils.py`

```python
# Current scattered functions
def handle_tools(response_model, kwargs):
    ...

def handle_json_modes(response_model, kwargs, mode):
    ...

def reask_tools(kwargs, response, exception):
    ...
```

**New**: `instructor/providers/openai/mode_handlers.py` (new file)

```python
from instructor.mode import Mode
from instructor.core.mode_registry import register_mode_handler
from instructor.core.base_mode_handler import BaseModeHandler
from instructor.processing.function_calls import convert_to_openai_tool
from typing import Any, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

@register_mode_handler(Mode.TOOLS, lazy=True)
class OpenAIToolsHandler(BaseModeHandler):
    """Handler for OpenAI tools mode."""

    def __init__(self):
        super().__init__(provider="openai")

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_parallel(self) -> bool:
        return False

    def prepare_request(
        self,
        response_model: type[T],
        kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Add tools to kwargs."""
        kwargs_copy = kwargs.copy()

        # Convert model to OpenAI tool format
        tool = convert_to_openai_tool(response_model)
        kwargs_copy["tools"] = [tool]
        kwargs_copy["tool_choice"] = {
            "type": "function",
            "function": {"name": tool["function"]["name"]}
        }

        return response_model, kwargs_copy

    def process_response(
        self,
        response: Any,
        response_model: type[T],
        **kwargs: Any
    ) -> T:
        """Extract model from tool call."""
        from instructor.processing.response import extract_tool_call

        tool_call = extract_tool_call(response)
        return response_model.model_validate_json(tool_call.function.arguments)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception
    ) -> dict[str, Any]:
        """Append validation error to messages."""
        kwargs_copy = kwargs.copy()
        messages = kwargs_copy.get("messages", [])

        # Add tool response with error
        messages.append({
            "role": "tool",
            "tool_call_id": response.choices[0].message.tool_calls[0].id,
            "content": f"Validation error: {exception}"
        })

        kwargs_copy["messages"] = messages
        return kwargs_copy

@register_mode_handler(Mode.JSON, lazy=True)
class OpenAIJSONHandler(BaseModeHandler):
    """Handler for OpenAI JSON mode."""

    def __init__(self):
        super().__init__(provider="openai")

    @property
    def supports_streaming(self) -> bool:
        return True

    def prepare_request(
        self,
        response_model: type[T],
        kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Add response_format to kwargs."""
        kwargs_copy = kwargs.copy()
        kwargs_copy["response_format"] = {"type": "json_object"}
        return response_model, kwargs_copy

    def process_response(
        self,
        response: Any,
        response_model: type[T],
        **kwargs: Any
    ) -> T:
        """Parse JSON from content."""
        content = response.choices[0].message.content
        return response_model.model_validate_json(content)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception
    ) -> dict[str, Any]:
        """Standard reask - append error to messages."""
        return super().handle_reask(kwargs, response, exception)

# Register remaining OpenAI modes
# TOOLS_STRICT, PARALLEL_TOOLS, JSON_O1, etc.
```

**Import in `__init__.py`**:

```python
# instructor/providers/openai/__init__.py

# Import to trigger registration (lazy loading)
from . import mode_handlers  # noqa: F401
```

**Test**:
```python
# tests/providers/test_openai_handlers.py
from instructor.mode import Mode
from instructor.core.mode_registry import mode_registry
from pydantic import BaseModel

def test_openai_tools_registered():
    """Test that OpenAI handlers are registered."""
    assert mode_registry.has_mode(Mode.TOOLS)
    assert mode_registry.has_mode(Mode.JSON)

def test_openai_tools_handler():
    """Test OpenAI tools handler."""
    handler = mode_registry.get_handler(Mode.TOOLS)

    assert handler.provider == "openai"
    assert handler.supports_streaming

    class User(BaseModel):
        name: str

    response_model, kwargs = handler.prepare_request(User, {})
    assert "tools" in kwargs
    assert kwargs["tool_choice"]["type"] == "function"
```

---

### Step 3: Update response.py to Use Registry (Week 3)

**Current**: `instructor/processing/response.py`

```python
# OLD CODE (lines 432-469)
mode_handlers = {
    Mode.TOOLS: handle_tools,
    Mode.JSON: lambda rm, nk: handle_json_modes(rm, nk, Mode.JSON),
    # ... 32 more
}

if mode in mode_handlers:
    response_model, new_kwargs = mode_handlers[mode](response_model, new_kwargs)
else:
    raise ValueError(f"Invalid patch mode: {mode}")
```

**New**:
```python
# NEW CODE
from instructor.core.mode_registry import mode_registry

def handle_response_model(
    response_model: type[BaseModel],
    mode: Mode,
    **kwargs
) -> tuple[type[BaseModel], dict[str, Any]]:
    """Prepare response model for API request."""

    # Get handler from registry
    try:
        handler = mode_registry.get_handler(mode)
    except ValueError as e:
        # Better error message
        available_modes = mode_registry.list_modes()
        raise ValueError(
            f"Unsupported mode: {mode}. "
            f"Available modes: {', '.join(str(m) for m in available_modes)}"
        ) from e

    # Use handler
    return handler.prepare_request(response_model, kwargs)
```

**Similarly for reask**:
```python
# OLD CODE (lines 612-663)
REASK_HANDLERS = {
    Mode.TOOLS: reask_tools,
    # ... 36 more
}

if mode in REASK_HANDLERS:
    return REASK_HANDLERS[mode](kwargs_copy, response, exception)
else:
    return reask_default(kwargs_copy, response, exception)
```

**New**:
```python
def handle_reask_kwargs(
    mode: Mode,
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception
) -> dict[str, Any]:
    """Handle reask for validation failure."""

    # Get handler from registry
    handler = mode_registry.get_handler(mode)

    # Use handler
    return handler.handle_reask(kwargs, response, exception)
```

**Keep Old Code Initially** (backward compatibility):

```python
# TRANSITIONAL CODE - support both old and new

# Feature flag
USE_MODE_REGISTRY = os.getenv("INSTRUCTOR_USE_MODE_REGISTRY", "false").lower() == "true"

if USE_MODE_REGISTRY:
    # New registry-based code
    handler = mode_registry.get_handler(mode)
    response_model, new_kwargs = handler.prepare_request(response_model, kwargs)
else:
    # Old dict-based code (fallback)
    if mode in mode_handlers:
        response_model, new_kwargs = mode_handlers[mode](response_model, new_kwargs)
    else:
        raise ValueError(f"Invalid patch mode: {mode}")
```

---

### Step 4: Migrate Remaining Providers (Weeks 4-5)

**Priority Order**:
1. ✅ OpenAI (Week 2 - done)
2. Anthropic (Week 4, Day 1-2)
3. Gemini/VertexAI (Week 4, Day 3-5)
4. Mistral, Cohere (Week 5, Day 1-2)
5. All others (Week 5, Day 3-5)

**For each provider**:

1. Create `instructor/providers/{provider}/mode_handlers.py`
2. Create handler classes for each mode
3. Use `@register_mode_handler` decorator
4. Import in provider's `__init__.py`
5. Test with existing provider tests

**Example for Anthropic**:

```python
# instructor/providers/anthropic/mode_handlers.py

from instructor.mode import Mode
from instructor.core.mode_registry import register_mode_handler
from instructor.core.base_mode_handler import BaseModeHandler

@register_mode_handler(Mode.ANTHROPIC_TOOLS, lazy=True)
class AnthropicToolsHandler(BaseModeHandler):
    def __init__(self):
        super().__init__(provider="anthropic")

    @property
    def supports_streaming(self) -> bool:
        return True

    def prepare_request(self, response_model, kwargs):
        # Current logic from handle_anthropic_tools
        from ..utils import handle_anthropic_tools
        return handle_anthropic_tools(response_model, kwargs)

    def process_response(self, response, response_model, **kwargs):
        # Extract from tool use
        ...

    def handle_reask(self, kwargs, response, exception):
        # Current logic from reask_anthropic_tools
        from ..utils import reask_anthropic_tools
        return reask_anthropic_tools(kwargs, response, exception)

# Register other Anthropic modes
```

**Test each provider**:
```bash
# After migrating Anthropic
INSTRUCTOR_USE_MODE_REGISTRY=true uv run pytest tests/llm/test_anthropic/ -v

# After migrating Gemini
INSTRUCTOR_USE_MODE_REGISTRY=true uv run pytest tests/llm/test_gemini/ -v
```

---

### Step 5: Remove Old Dispatch Code (Week 6)

**After all providers migrated**:

1. Remove `mode_handlers` dictionary (lines 432-469)
2. Remove `REASK_HANDLERS` dictionary (lines 612-663)
3. Remove `PARALLEL_MODES` dictionary (lines 405-409)
4. Remove feature flag, make registry the default
5. Remove old imports from provider utils (lines 59-161)

**Before**:
```python
# instructor/processing/response.py (lines 59-161)
from ..providers.anthropic.utils import (
    handle_anthropic_json,
    handle_anthropic_parallel_tools,
    # ... 100+ lines of imports
)

mode_handlers = {  # lines 432-469
    Mode.TOOLS: handle_tools,
    # ... 34 entries
}

REASK_HANDLERS = {  # lines 612-663
    Mode.TOOLS: reask_tools,
    # ... 37 entries
}
```

**After**:
```python
# instructor/processing/response.py
from instructor.core.mode_registry import mode_registry

# That's it! No more imports needed
# Handlers are registered by providers themselves
```

**Line reduction**: ~180 lines removed

---

### Step 6: Documentation & Migration Guide (Week 6)

**Create documentation**:

1. **User Guide**: How to use the registry
2. **Provider Guide**: How to add new modes
3. **Migration Guide**: Updating custom modes

**File**: `docs/concepts/mode_registry.md`

```markdown
# Mode Registry

The Mode Registry is the central system for managing mode handlers in Instructor.

## For Users

When you use Instructor, modes are automatically registered by providers:

```python
import instructor
from instructor import Mode

# Modes are ready to use
client = instructor.from_openai(openai.OpenAI(), mode=Mode.TOOLS)
```

## For Provider Developers

### Adding a New Mode

Create a mode handler class:

```python
from instructor.core.mode_registry import register_mode_handler
from instructor.core.base_mode_handler import BaseModeHandler
from instructor.mode import Mode

@register_mode_handler(Mode.YOUR_MODE, lazy=True)
class YourModeHandler(BaseModeHandler):
    def __init__(self):
        super().__init__(provider="your_provider")

    def prepare_request(self, response_model, kwargs):
        # Add your provider-specific parameters
        return response_model, kwargs

    def process_response(self, response, response_model, **kwargs):
        # Extract and validate the model
        return validated_model

    def handle_reask(self, kwargs, response, exception):
        # Handle validation failures
        return updated_kwargs
```

Then import your handler module to trigger registration:

```python
# instructor/providers/your_provider/__init__.py
from . import mode_handlers  # Triggers registration
```

### Querying Registry

```python
from instructor.core.mode_registry import mode_registry

# Check if mode exists
if mode_registry.has_mode(Mode.TOOLS):
    handler = mode_registry.get_handler(Mode.TOOLS)

# List all modes
all_modes = mode_registry.list_modes()

# Get stats
stats = mode_registry.stats()
# {'loaded': 5, 'lazy': 32, 'total': 37}
```
```

---

## Testing Strategy

### Unit Tests

**Test the registry itself**:
```bash
uv run pytest tests/core/test_mode_registry.py -v
```

**Test each handler**:
```bash
uv run pytest tests/providers/test_openai_handlers.py -v
uv run pytest tests/providers/test_anthropic_handlers.py -v
```

### Integration Tests

**Test with actual API calls**:
```bash
# Enable registry
export INSTRUCTOR_USE_MODE_REGISTRY=true

# Run provider tests
uv run pytest tests/llm/test_openai/ -v
uv run pytest tests/llm/test_anthropic/ -v
```

### Regression Tests

**Compare old vs new**:
```python
# tests/regression/test_mode_registry_parity.py

def test_mode_registry_parity():
    """Ensure registry produces same results as old dicts."""

    # Test with old code
    old_result = old_handle_response_model(...)

    # Test with new registry
    new_result = new_handle_response_model(...)

    assert old_result == new_result
```

### Performance Tests

**Measure lazy loading benefit**:
```python
import time
from instructor.core.mode_registry import mode_registry

# Measure registry lookup time
start = time.time()
for _ in range(1000):
    handler = mode_registry.get_handler(Mode.TOOLS)
elapsed = time.time() - start

assert elapsed < 0.01  # Should be very fast
```

---

## Rollback Plan

### If Registry Has Issues

**Step 1**: Revert to feature flag
```python
# Set environment variable
export INSTRUCTOR_USE_MODE_REGISTRY=false
```

**Step 2**: If needed, revert commits
```bash
git revert <registry-commit-hash>
```

**Step 3**: Old code still works
- Dispatch dictionaries preserved during migration
- Feature flag controls which path is used

### If Specific Provider Has Issues

**Unregister the problematic mode**:
```python
from instructor.core.mode_registry import mode_registry

mode_registry.unregister(Mode.PROBLEMATIC_MODE)

# Fall back to old handler temporarily
```

---

## Success Criteria

Phase 1 is **complete** when:

- ☐ ModeRegistry implemented and tested
- ☐ ModeHandler protocol defined
- ☐ All 37 modes migrated to registry
- ☐ All provider tests passing with registry enabled
- ☐ Old dispatch code removed
- ☐ Import time <100ms (measured)
- ☐ Type checker passes without `# type: ignore`
- ☐ Documentation complete
- ☐ No regressions in functionality

---

## Metrics

**Before**:
- 3 dispatch dictionaries (74 entries total)
- 103 lines of provider imports
- Type safety defeated (`# type: ignore`)
- ~500ms import time

**After**:
- 1 registry (~150 lines)
- 0 provider imports in response.py
- Full type safety
- <100ms import time (90% faster)

**Lines Eliminated**: ~180 lines from response.py

---

## Next Phase

After Phase 1 completes, proceed to:
- **[Phase 2: Provider Registry](./phase2_provider_registry.md)** - Eliminate auto_client.py if/elif chain
- **[Phase 3: Lazy Loading](./phase3_lazy_loading.md)** - Full lazy loading implementation

---

**Status**: Ready to start (Theme 1 Phase 2 recommended first)
**Dependencies**: None (but Base Classes from Theme 1 Phase 2 helps)
**Timeline**: 4-6 weeks
