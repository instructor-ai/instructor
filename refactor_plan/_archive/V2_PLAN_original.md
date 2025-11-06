# Instructor 2.0 Roadmap: Incremental Path to Excellence

This document outlines a pragmatic, incremental approach to evolving Instructor toward version 2.0. Rather than a big-bang refactor, we'll introduce improvements progressively while maintaining backward compatibility.

---

## Executive Summary

**Current State**: Solid foundation with 16 providers, but suffering from:

- 40+ mode dispatcher with tight coupling (`processing/response.py:430-467`)
- 3,486 lines of duplicated provider code across 11 files
- Giant if/elif chain for provider detection (`auto_client.py:157-700+`)
- Sync/async duplication in retry logic (`core/retry.py:143-450`)

**Vision for 2.0**:

- **Decoupled**: Providers as plugins, not hardcoded dependencies
- **Extensible**: Registry-based architecture for modes and providers
- **Performant**: Lazy loading, cached schemas, streaming optimization
- **Modern**: Python 3.10+, full Pydantic 2, clean type system

---

## Part A: Quality of Life Improvements (Parallelizable)

These tasks can be implemented **independently and in parallel**. They improve developer experience, performance, and code quality without requiring architectural changes.

### A.1 Resolve TODOs and Technical Debt (1-2 weeks)

**Impact**: Code quality, maintainability  
**Effort**: Low  
**Risk**: Minimal  
**Can parallelize**: Yes (each TODO is independent)

**Action Items**:

1. **Fix content type handling in function_calls.py**
   - Location: `instructor/processing/function_calls.py:302, 340`
   - Current: `# TODO: Handle other content types`
   - Fix: Add proper handling for image/audio content in Anthropic schemas

2. **Stabilize Anthropic Batch API**
   - Locations: 7 TODOs across `instructor/batch/` files
   - Current: Beta API workarounds with `TODO(#batch-api-stable)`
   - Fix: Update to stable batch API once available

3. **Fix VertexAI Iterable fields**
   - Location: `instructor/providers/vertexai/client.py:25`
   - Current: `# TODO: Figure out why vertexai needs required fields`
   - Fix: Investigate and document the root cause

**Files to modify**:
```
instructor/processing/function_calls.py
instructor/batch/providers/anthropic.py
instructor/providers/vertexai/client.py
```

### A.2 Add Schema Caching (1 week)

**Impact**: Performance improvement (30-50% faster for repeated calls)  
**Effort**: Low  
**Risk**: Minimal  
**Can parallelize**: Yes (independent feature)

**Location**: `instructor/processing/function_calls.py`

```python
from functools import lru_cache
from typing import Any
from pydantic import BaseModel

@lru_cache(maxsize=256)
def get_cached_schema(
    model_name: str,
    model_hash: int,
    mode: str
) -> dict[str, Any]:
    """Cache generated schemas to avoid repeated computation."""
    # Schema generation is deterministic for a given model
    pass

def model_to_json_schema(
    model: type[BaseModel],
    mode: Mode
) -> dict[str, Any]:
    """Generate JSON schema with caching."""
    model_hash = hash(frozenset(model.model_fields.keys()))
    cache_key = (model.__name__, model_hash, mode.value)
    
    # Check cache first
    return get_cached_schema(*cache_key)
```

**Benefits**:
- Significant performance boost for repeated calls
- No API changes
- Simple implementation

### A.3 Improve Error Messages (1-2 weeks)

**Impact**: Developer experience  
**Effort**: Low  
**Risk**: Minimal  
**Can parallelize**: Yes (can be done incrementally)

**Current issues**:
```python
# Too generic
raise ValueError(f"Invalid patch mode: {mode}")

# Missing context
raise ConfigurationError("API key required")
```

**Improved**:
```python
raise ValueError(
    f"Invalid patch mode: {mode.value}. "
    f"Supported modes for {provider}: {', '.join(supported_modes)}"
)

raise ConfigurationError(
    f"API key required for {provider}. "
    f"Set environment variable {env_var} or pass api_key parameter. "
    f"See: https://docs.instructor.com/integrations/{provider}"
)
```

**Action**: Create `instructor/core/errors.py` with helper functions for rich error messages.

### A.4 Add Type Stubs and Remove type: ignore (2 weeks)

**Impact**: Better IDE support, type safety  
**Effort**: Medium  
**Risk**: Minimal  
**Can parallelize**: Yes (can be done file-by-file)

**Current issues**:
- `# type: ignore` scattered throughout codebase
- Missing type stubs for external dependencies
- Generic `Any` types where specific types possible

**Action**:
1. Run `uv run ty check` to find all type errors
2. Add proper types instead of `# type: ignore`
3. Create type stubs for untyped dependencies
4. Enable stricter type checking in `pyproject.toml`

### A.5 Consolidate Retry Logic (2-3 weeks)

**Impact**: Eliminate sync/async duplication  
**Effort**: Medium  
**Risk**: Low  
**Can parallelize**: Yes (refactoring task, doesn't affect API)

**Current problem** (`core/retry.py:143-450`):
- `retry_sync`: 153 lines
- `retry_async`: 150 lines (nearly identical)

**Solution**: Extract common logic

```python
# instructor/core/retry_core.py
from typing import TypeVar, Callable, Any, Coroutine
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class RetryContext:
    """Shared context for retry operations."""
    response_model: type[T] | None
    max_retries: int
    strict: bool | None
    mode: Mode
    hooks: Hooks
    total_usage: Any
    failed_attempts: list[FailedAttempt]

def create_retry_context(
    response_model: type[T] | None,
    max_retries: int | Retrying,
    strict: bool | None,
    mode: Mode,
    hooks: Hooks | None,
    is_async: bool = False
) -> tuple[RetryContext, Retrying]:
    """Create shared retry context."""
    hooks = hooks or Hooks()
    total_usage = initialize_usage(mode)
    max_retries_obj = initialize_retrying(max_retries, is_async=is_async)
    
    return RetryContext(
        response_model=response_model,
        max_retries=max_retries_obj,
        strict=strict,
        mode=mode,
        hooks=hooks,
        total_usage=total_usage,
        failed_attempts=[]
    ), max_retries_obj

def handle_retry_exception(
    context: RetryContext,
    exception: Exception,
    attempt_number: int
) -> None:
    """Common exception handling for retries."""
    logger.debug(f"Parse error: {exception}")
    context.hooks.emit_parse_error(exception)
    
    context.failed_attempts.append(
        FailedAttempt(
            attempt_number=attempt_number,
            exception=exception,
        )
    )

# Then retry_sync and retry_async just call these helpers
```

**Benefits**:
- ~300 lines of duplication eliminated
- Single source of truth for retry logic
- Easier to test and maintain

### A.6 Lazy Provider Loading (2-3 weeks)

**Impact**: Faster import time, smaller memory footprint  
**Effort**: Medium  
**Risk**: Low  
**Can parallelize**: Yes (infrastructure change, doesn't affect API)

**Current problem**: Importing `instructor` loads all provider modules

**Solution**: `instructor/providers/__init__.py`

```python
from typing import TYPE_CHECKING
import importlib
from functools import lru_cache

# Only import types, not implementations
if TYPE_CHECKING:
    from .anthropic.client import from_anthropic
    from .openai.client import from_openai
    # ... etc

@lru_cache(maxsize=None)
def get_provider_module(provider: str):
    """Lazily import provider module."""
    try:
        return importlib.import_module(f"instructor.providers.{provider}")
    except ImportError:
        raise ImportError(f"Provider {provider} not available")

def __getattr__(name: str):
    """Lazy loading of provider functions."""
    # Map function names to provider modules
    provider_map = {
        'from_anthropic': 'anthropic',
        'from_openai': 'openai',
        'from_gemini': 'gemini',
        # ... etc
    }
    
    if name in provider_map:
        module = get_provider_module(provider_map[name])
        return getattr(module.client, name)
    
    raise AttributeError(f"module 'instructor.providers' has no attribute '{name}'")
```

**Benefits**:
- Import time: ~500ms → ~50ms (estimated)
- Memory: Only load providers actually used
- No API changes

### A.7 Streaming Optimization (2-3 weeks)

**Impact**: Better performance for streaming responses  
**Effort**: Medium  
**Risk**: Low  
**Can parallelize**: Yes (optimization task)

**Current issues**:
- Buffer management in Partial responses
- Type manipulation overhead
- Repeated validation

**Solutions**:

```python
# instructor/dsl/streaming.py
class StreamBuffer:
    """Optimized buffer for streaming responses."""
    
    __slots__ = ['_chunks', '_model', '_validator', '_cache']
    
    def __init__(self, model: type[BaseModel]):
        self._chunks: list[str] = []
        self._model = model
        self._validator = model.__pydantic_validator__
        self._cache: dict[int, Any] = {}
    
    def add_chunk(self, chunk: str) -> BaseModel | None:
        """Add chunk and attempt validation."""
        self._chunks.append(chunk)
        content = ''.join(self._chunks)
        
        # Hash content to check cache
        content_hash = hash(content)
        if content_hash in self._cache:
            return self._cache[content_hash]
        
        try:
            result = self._validator.validate_json(content)
            self._cache[content_hash] = result
            return result
        except ValidationError:
            return None
```

### A.8 Parallel Processing Utilities (2-3 weeks)

**Impact**: Better batch processing capabilities  
**Effort**: Medium  
**Risk**: Low  
**Can parallelize**: Yes (new feature, doesn't affect existing code)

**Vision**: Batch multiple requests efficiently

```python
# instructor/batch/parallel.py
import asyncio
from typing import TypeVar, Generic

T = TypeVar('T', bound=BaseModel)

class ParallelProcessor(Generic[T]):
    """Process multiple requests in parallel."""
    
    def __init__(
        self,
        client: AsyncInstructor,
        max_concurrency: int = 10
    ):
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_batch(
        self,
        prompts: list[str],
        response_model: type[T],
        **kwargs
    ) -> list[T]:
        """Process batch of prompts in parallel."""
        async def process_one(prompt: str) -> T:
            async with self.semaphore:
                return await self.client.create(
                    messages=[{"role": "user", "content": prompt}],
                    response_model=response_model,
                    **kwargs
                )
        
        return await asyncio.gather(*[
            process_one(prompt) for prompt in prompts
        ])

# Usage:
processor = ParallelProcessor(client, max_concurrency=50)
results = await processor.process_batch(
    prompts=["prompt 1", "prompt 2", ...],
    response_model=MyModel
)
```

---

## Part B: Large Architectural Redesign

These tasks require coordinated changes across the codebase and introduce new architectural patterns. They should be done sequentially or with careful coordination.

### B.1 Provider Base Classes (2-3 weeks)

**Impact**: Foundation for reducing code duplication  
**Effort**: Medium  
**Risk**: Low (new code, doesn't break existing)  
**Dependencies**: None (can start first)

**Create**: `instructor/providers/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class BaseProviderHandler(ABC):
    """Base class for provider-specific request/response handling."""
    
    @abstractmethod
    def handle_tools_mode(
        self,
        response_model: type[T],
        kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Prepare request for tools mode."""
        pass
    
    @abstractmethod
    def handle_json_mode(
        self,
        response_model: type[T],
        kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Prepare request for JSON mode."""
        pass
    
    @abstractmethod
    def reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception
    ) -> dict[str, Any]:
        """Handle reask logic for validation failures."""
        pass
    
    def format_error_message(self, exception: Exception) -> str:
        """Format validation error for reask. Override for custom formatting."""
        return str(exception)
```

**Benefits**:
- Enforces consistent interface across providers
- Reduces duplication (3,486 lines → estimated 1,500 lines)
- Makes adding new providers easier
- Can be adopted incrementally (doesn't break existing code)

**Migration strategy**:
1. Create base class
2. Refactor one provider (e.g., Anthropic) to inherit from base
3. Gradually migrate other providers
4. Keep old code until all providers migrated

### B.2 Mode Handler Registry (4-6 weeks)

**Impact**: Decouple mode handling, enable dynamic registration  
**Effort**: High  
**Risk**: Medium (requires careful testing)  
**Dependencies**: B.1 (Provider Base Classes) helpful but not required

**Current problem** (`processing/response.py:430-467`):

```python
mode_handlers = {
    Mode.TOOLS: handle_tools,
    Mode.MISTRAL_TOOLS: handle_mistral_tools,
    Mode.ANTHROPIC_TOOLS: handle_anthropic_tools,
    # ... 34 more hardcoded handlers
}
```

**New design**: `instructor/core/mode_registry.py`

```python
from typing import Protocol, Callable, Any
from enum import Enum
from pydantic import BaseModel

class ModeHandler(Protocol):
    """Protocol for mode handler functions."""
    
    def prepare_request(
        self,
        response_model: type[BaseModel],
        kwargs: dict[str, Any]
    ) -> tuple[type[BaseModel], dict[str, Any]]:
        """Prepare request for this mode."""
        ...
    
    def process_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """Process response for this mode."""
        ...
    
    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception
    ) -> dict[str, Any]:
        """Handle validation failure reask."""
        ...

class ModeRegistry:
    """Registry for mode handlers with lazy loading."""
    
    def __init__(self):
        self._handlers: dict[Mode, ModeHandler] = {}
        self._lazy_loaders: dict[Mode, Callable[[], ModeHandler]] = {}
    
    def register(
        self,
        mode: Mode,
        handler: ModeHandler | None = None,
        lazy_loader: Callable[[], ModeHandler] | None = None
    ):
        """Register a handler for a mode.
        
        Args:
            mode: The mode to register
            handler: Pre-instantiated handler (loaded immediately)
            lazy_loader: Function to create handler (loaded on first use)
        """
        if handler:
            self._handlers[mode] = handler
        elif lazy_loader:
            self._lazy_loaders[mode] = lazy_loader
        else:
            raise ValueError("Must provide handler or lazy_loader")
    
    def get_handler(self, mode: Mode) -> ModeHandler:
        """Get handler for mode, loading lazily if needed."""
        if mode in self._handlers:
            return self._handlers[mode]
        
        if mode in self._lazy_loaders:
            # Load on first use
            handler = self._lazy_loaders[mode]()
            self._handlers[mode] = handler
            del self._lazy_loaders[mode]  # Free memory
            return handler
        
        raise ValueError(f"No handler registered for mode: {mode}")

# Global registry
mode_registry = ModeRegistry()

# Decorator for easy registration
def register_mode_handler(mode: Mode):
    """Decorator to register a mode handler."""
    def decorator(handler_class):
        mode_registry.register(
            mode,
            lazy_loader=lambda: handler_class()
        )
        return handler_class
    return decorator
```

**Usage in provider files**:

```python
# instructor/providers/anthropic/handlers.py
from instructor.core.mode_registry import register_mode_handler, ModeHandler
from instructor.mode import Mode

@register_mode_handler(Mode.ANTHROPIC_TOOLS)
class AnthropicToolsHandler(ModeHandler):
    def prepare_request(self, response_model, kwargs):
        # Current logic from handle_anthropic_tools
        return response_model, kwargs
    
    def process_response(self, response, response_model, **kwargs):
        # Response processing logic
        return response
    
    def handle_reask(self, kwargs, response, exception):
        # Current logic from reask_anthropic_tools
        return kwargs
```

**Migration strategy**:
1. Create registry infrastructure
2. Migrate one mode (e.g., ANTHROPIC_TOOLS) to new pattern
3. Support both old dict-based and new registry-based lookup
4. Gradually migrate all modes
5. Deprecate old pattern in v1.15
6. Remove in v2.0

**Benefits**:
- Providers can register modes independently
- Lazy loading reduces import time
- Easier to add new modes
- Testable in isolation

### B.3 Provider Registry (4-6 weeks)

**Impact**: Eliminate giant if/elif chain, enable provider plugins  
**Effort**: High  
**Risk**: Medium  
**Dependencies**: B.2 (Mode Handler Registry) helpful but not required

**Current problem** (`auto_client.py:157-700`):

```python
if provider == "openai":
    # 40+ lines of setup
elif provider == "anthropic":
    # 40+ lines of nearly identical setup
elif provider == "gemini":
    # ... repeat 14 more times
```

**New design**: `instructor/core/provider_registry.py`

```python
from typing import Protocol, Any, Callable
from dataclasses import dataclass
from instructor.mode import Mode

@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    name: str
    package_name: str  # For import checking
    default_mode: Mode
    supported_modes: list[Mode]
    env_var_name: str | None = None
    docs_url: str | None = None

class ProviderFactory(Protocol):
    """Protocol for provider factory functions."""
    
    def create_client(
        self,
        api_key: str | None = None,
        model: str | None = None,
        mode: Mode | None = None,
        async_client: bool = False,
        **kwargs: Any
    ) -> Any:
        """Create an instructor client for this provider."""
        ...

class ProviderRegistry:
    """Registry for provider factories."""
    
    def __init__(self):
        self._factories: dict[str, tuple[ProviderConfig, ProviderFactory]] = {}
    
    def register(
        self,
        config: ProviderConfig,
        factory: ProviderFactory
    ):
        """Register a provider factory."""
        self._factories[config.name] = (config, factory)
    
    def create_client(
        self,
        provider: str,
        **kwargs
    ) -> Any:
        """Create a client for the specified provider."""
        if provider not in self._factories:
            available = ', '.join(self._factories.keys())
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {available}"
            )
        
        config, factory = self._factories[provider]
        
        # Check if package is installed
        try:
            __import__(config.package_name)
        except ImportError:
            raise ConfigurationError(
                f"The {config.package_name} package is required for {provider}. "
                f"Install it with: pip install {config.package_name}"
                f"\nSee: {config.docs_url or 'https://docs.instructor.com'}"
            )
        
        return factory.create_client(**kwargs)
    
    def list_providers(self) -> list[str]:
        """List all registered providers."""
        return list(self._factories.keys())

# Global registry
provider_registry = ProviderRegistry()

# Decorator for registration
def register_provider(config: ProviderConfig):
    """Decorator to register a provider factory."""
    def decorator(factory_func):
        provider_registry.register(config, factory_func)
        return factory_func
    return decorator
```

**Usage in provider files**:

```python
# instructor/providers/anthropic/factory.py
from instructor.core.provider_registry import register_provider, ProviderConfig
from instructor.mode import Mode

ANTHROPIC_CONFIG = ProviderConfig(
    name="anthropic",
    package_name="anthropic",
    default_mode=Mode.ANTHROPIC_TOOLS,
    supported_modes=[
        Mode.ANTHROPIC_TOOLS,
        Mode.ANTHROPIC_JSON,
        Mode.ANTHROPIC_REASONING_TOOLS,
        Mode.ANTHROPIC_PARALLEL_TOOLS,
    ],
    env_var_name="ANTHROPIC_API_KEY",
    docs_url="https://docs.instructor.com/integrations/anthropic"
)

@register_provider(ANTHROPIC_CONFIG)
def create_anthropic_client(
    api_key: str | None = None,
    model: str | None = None,
    mode: Mode | None = None,
    async_client: bool = False,
    **kwargs
):
    """Factory for Anthropic instructor clients."""
    import anthropic
    from instructor import from_anthropic
    
    client = (
        anthropic.AsyncAnthropic(api_key=api_key)
        if async_client
        else anthropic.Anthropic(api_key=api_key)
    )
    
    return from_anthropic(
        client,
        mode=mode or ANTHROPIC_CONFIG.default_mode,
        **kwargs
    )
```

**New auto_client.py** becomes:

```python
def from_provider(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
    **kwargs
):
    """Create instructor client from provider name."""
    return provider_registry.create_client(
        provider=provider,
        api_key=api_key,
        model=model,
        **kwargs
    )
```

**Benefits**:
- 1,080 lines → ~100 lines
- Providers can be added without modifying core
- Plugin architecture for future extensions
- Consistent error handling

### B.4 Hierarchical Mode System (6-8 weeks)

**Impact**: Rich mode metadata, better organization  
**Effort**: High  
**Risk**: Medium (breaking change)  
**Dependencies**: B.2 (Mode Handler Registry)  
**Timeline**: v2.0 breaking change

**Vision**: Modes organized by provider and capability

```python
# instructor/mode_v2.py
from enum import Enum
from dataclasses import dataclass
from typing import Protocol

class ModeCapability(Enum):
    """Base capabilities modes can have."""
    TOOLS = "tools"
    JSON = "json"
    STREAMING = "streaming"
    PARALLEL = "parallel"
    REASONING = "reasoning"

@dataclass
class ModeDescriptor:
    """Rich metadata for a mode."""
    provider: str
    capabilities: set[ModeCapability]
    supports_streaming: bool
    supports_vision: bool
    supports_audio: bool
    api_field_name: str | None = None

class Mode(Enum):
    """Hierarchical mode system."""
    
    # OpenAI modes
    OPENAI_TOOLS = ModeDescriptor(
        provider="openai",
        capabilities={ModeCapability.TOOLS, ModeCapability.STREAMING},
        supports_streaming=True,
        supports_vision=True,
        supports_audio=False,
        api_field_name="tools"
    )
    
    OPENAI_JSON = ModeDescriptor(
        provider="openai",
        capabilities={ModeCapability.JSON, ModeCapability.STREAMING},
        supports_streaming=True,
        supports_vision=True,
        supports_audio=False,
        api_field_name="response_format"
    )
    
    # Anthropic modes
    ANTHROPIC_TOOLS = ModeDescriptor(
        provider="anthropic",
        capabilities={ModeCapability.TOOLS, ModeCapability.STREAMING},
        supports_streaming=True,
        supports_vision=True,
        supports_audio=False,
        api_field_name="tools"
    )
    
    # ... etc
    
    @property
    def provider(self) -> str:
        return self.value.provider
    
    @property
    def supports_streaming(self) -> bool:
        return self.value.supports_streaming
    
    def has_capability(self, capability: ModeCapability) -> bool:
        return capability in self.value.capabilities
```

**Benefits**:
- Rich mode metadata
- Easy to query capabilities
- Provider-aware
- Self-documenting

### B.5 Plugin Architecture (8-10 weeks)

**Impact**: Third-party providers without core modifications  
**Effort**: High  
**Risk**: Medium  
**Dependencies**: B.2 (Mode Handler Registry), B.3 (Provider Registry)  
**Timeline**: v2.0 feature

**Vision**: Third-party providers without core modifications

```python
# instructor/plugin_api.py
from typing import Protocol
from instructor.mode import Mode
from instructor.core.mode_registry import ModeHandler

class InstructorPlugin(Protocol):
    """Protocol for Instructor plugins."""
    
    name: str
    version: str
    modes: list[Mode]
    
    def register(self, registry) -> None:
        """Register plugin with instructor."""
        ...
    
    def create_client(self, **kwargs) -> Any:
        """Create client for this plugin."""
        ...

# Third-party provider example:
# instructor-ollama package
class OllamaPlugin:
    name = "ollama"
    version = "1.0.0"
    modes = [Mode.OLLAMA_JSON, Mode.OLLAMA_TOOLS]
    
    def register(self, registry):
        from .handlers import OllamaToolsHandler, OllamaJSONHandler
        registry.register_mode(Mode.OLLAMA_TOOLS, OllamaToolsHandler())
        registry.register_mode(Mode.OLLAMA_JSON, OllamaJSONHandler())
    
    def create_client(self, **kwargs):
        from .client import from_ollama
        return from_ollama(**kwargs)

# Usage:
import instructor
from instructor_ollama import OllamaPlugin

instructor.register_plugin(OllamaPlugin())
client = instructor.from_provider("ollama")
```

**Benefits**:
- Extensible without modifying core
- Third-party provider ecosystem
- Clear plugin API

### B.6 Configuration System (4-6 weeks)

**Impact**: Centralized configuration instead of scattered kwargs  
**Effort**: Medium  
**Risk**: Low  
**Dependencies**: None  
**Timeline**: v2.0 feature

**Vision**: Centralized configuration instead of scattered kwargs

```python
# instructor/config.py
from dataclasses import dataclass
from typing import Any

@dataclass
class InstructorConfig:
    """Global instructor configuration."""
    
    # Retry settings
    max_retries: int = 3
    retry_on: list[type[Exception]] = None
    
    # Validation
    strict_validation: bool = False
    validation_context: dict[str, Any] = None
    
    # Hooks
    enable_hooks: bool = True
    log_level: str = "INFO"
    
    # Performance
    cache_schemas: bool = True
    lazy_load_providers: bool = True
    
    # Development
    debug_mode: bool = False
    raise_on_validation_error: bool = True
    
    @classmethod
    def from_env(cls) -> "InstructorConfig":
        """Load configuration from environment variables."""
        import os
        return cls(
            max_retries=int(os.getenv("INSTRUCTOR_MAX_RETRIES", "3")),
            strict_validation=os.getenv("INSTRUCTOR_STRICT", "false").lower() == "true",
            # ... etc
        )

# Usage:
import instructor

# Global config
instructor.configure(
    max_retries=5,
    cache_schemas=True
)

# Per-client config
client = instructor.from_openai(
    openai.OpenAI(),
    config=instructor.InstructorConfig(
        strict_validation=True,
        debug_mode=True
    )
)
```

**Benefits**:
- Discoverable configuration
- Type-safe settings
- Environment variable support
- Per-client overrides

### B.7 Modernize Type System (3-4 weeks)

**Impact**: Python 3.10+ syntax, cleaner code  
**Effort**: Medium  
**Risk**: Low (but requires Python 3.10+)  
**Dependencies**: None  
**Timeline**: v2.0 breaking change

**Changes for Python 3.10+**:

```python
# Before (Python 3.9 compatible)
from typing import Union, Optional
from __future__ import annotations

def process(value: Optional[Union[str, int]]) -> Union[str, None]:
    pass

# After (Python 3.10+)
def process(value: str | int | None) -> str | None:
    pass
```

**Remove compatibility code**:

```python
# Remove these
UNION_ORIGINS = (Union, types.UnionType) if hasattr(types, "UnionType") else (Union,)

# Use native syntax
if isinstance(origin, types.UnionType):
    ...
```

**Use match/case for mode dispatch**:

```python
# Before
if mode == Mode.TOOLS:
    handler = handle_tools
elif mode == Mode.ANTHROPIC_TOOLS:
    handler = handle_anthropic_tools
# ... 38 more

# After
match mode:
    case Mode.TOOLS:
        handler = handle_tools
    case Mode.ANTHROPIC_TOOLS:
        handler = handle_anthropic_tools
    case _:
        raise ValueError(f"Unknown mode: {mode}")
```

---

## Execution Strategy

### Parallel Workflow

**Quality of Life tasks (Part A)** can be worked on simultaneously:
- Multiple developers can pick different tasks
- No coordination needed
- Can be merged independently
- Immediate value delivery

**Architectural tasks (Part B)** require coordination:
- B.1 (Base Classes) can start first - foundation for others
- B.2 (Mode Registry) and B.3 (Provider Registry) benefit from B.1
- B.4-B.7 are v2.0 changes and can be planned for later

### Recommended Order

1. **Start Part A tasks immediately** (all in parallel)
   - A.1: Fix TODOs
   - A.2: Schema caching
   - A.3: Error messages
   - A.4: Type improvements

2. **Begin Part B foundation** (sequential)
   - B.1: Provider base classes (foundation)
   - Then Part A.5, A.6 can continue in parallel

3. **Build registries** (after B.1)
   - B.2: Mode registry
   - B.3: Provider registry

4. **v2.0 features** (timeline dependent)
   - B.4-B.7 can be planned for v2.0 release

---

## Estimated Timeline

| Category | Tasks | Duration | Effort (person-weeks) |
|----------|-------|----------|----------------------|
| **Part A: QoL** | A.1-A.8 | 2-3 months | 10-14 weeks |
| **Part B: Architecture** | B.1-B.3 | 4-6 months | 18-24 weeks |
| **Part B: v2.0** | B.4-B.7 | 6-9 months | 24-36 weeks |
| **Total** | | 12-18 months | ~60-80 weeks |

---

## Success Metrics

### Code Quality
- LoC reduction: 3,486 → ~1,500 (provider utils)
- Duplication: 40% → <10%
- Type coverage: 85% → 98%
- TODOs: 15 → 0

### Performance
- Import time: 500ms → 50ms
- Schema generation: Cache hit rate >80%
- Streaming throughput: +30%
- Memory usage: -20%

### Developer Experience
- Time to add new provider: 8 hours → 2 hours
- Time to add new mode: 4 hours → 30 minutes
- Documentation completeness: 70% → 95%
- Issue resolution time: -50%

### Community
- Third-party providers: 0 → 5+
- Plugin ecosystem established
- Migration guide completion rate: >90%

---

## Migration Strategy

### v1.14 - v1.15: Preparation Phase (3-6 months)

**Goals**: Introduce new patterns while maintaining compatibility

1. **Add new infrastructure** (no deprecations yet):
   - Base provider classes
   - Mode registry (alongside old dict dispatch)
   - Provider registry (alongside old if/elif)
   - Schema caching
   - Lazy loading

2. **Migrate internal code**:
   - Use new patterns internally
   - Keep old patterns as fallback
   - Comprehensive testing

3. **Documentation**:
   - Document new patterns
   - Migration guide
   - Code examples

### v1.16 - v1.17: Deprecation Phase (2-3 months)

**Goals**: Warn users about upcoming changes

1. **Deprecation warnings**:
   - Old import paths
   - Direct mode handler access
   - Provider-specific internal APIs

2. **Enhanced warnings**:

```python
import warnings

warnings.warn(
    "Importing from instructor.client is deprecated. "
    "Use from instructor.core.client instead. "
    "This will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2
)
```

3. **Automated migration tool**:

```bash
uv run instructor migrate --from 1.x --to 2.0 <directory>
```

### v2.0: Breaking Changes (1 month stabilization)

**Breaking changes**:
1. Remove old import paths
2. Python 3.10+ required
3. Pydantic 2.8+ required
4. New mode enum structure
5. Provider registration required for custom providers

**Compatibility layer**:

```python
# instructor/compat.py
"""Compatibility layer for 1.x code."""

def from_openai_v1(*args, **kwargs):
    """Deprecated: Use from_openai instead."""
    warnings.warn("Use from_openai", DeprecationWarning)
    from instructor import from_openai
    return from_openai(*args, **kwargs)
```

---

## Testing Strategy

### For Incremental Changes
1. **Regression tests**: Ensure old behavior still works
2. **Feature tests**: Test new patterns work correctly
3. **Integration tests**: Test old + new patterns together
4. **Performance tests**: Benchmark improvements

### For 2.0
1. **Comprehensive test suite**: Every provider, every mode
2. **Migration tests**: Automated migration tool validation
3. **Backward compatibility tests**: Ensure compat layer works
4. **Performance benchmarks**: Compare 1.x vs 2.0

---

## Risk Mitigation

### Technical Risks

1. **Breaking changes alienate users**
   - Mitigation: Long deprecation period (6+ months)
   - Automated migration tool
   - Comprehensive documentation

2. **Performance regressions**
   - Mitigation: Continuous benchmarking
   - Performance tests in CI
   - Canary releases

3. **Provider incompatibilities**
   - Mitigation: Extensive provider testing
   - Provider maintainer communication
   - Gradual rollout

### Process Risks

1. **Timeline slippage**
   - Mitigation: Incremental releases
   - Each phase independently valuable
   - Can skip/defer phases if needed

2. **Resource constraints**
   - Mitigation: Community contributions
   - Prioritize high-impact changes
   - Automated tooling where possible

---

## Recommendations for Next Steps

### Start Now (This Week)
1. **Fix TODOs** in `function_calls.py` and `batch/`
2. **Add schema caching** - quick win with immediate benefits
3. **Create base provider class** - doesn't break anything
4. **Improve error messages** - better DX immediately

### Next Month
1. **Prototype mode registry** - prove the concept
2. **Prototype provider registry** - prove the concept
3. **Write migration guide** - document the vision
4. **Get community feedback** - validate direction

### Next Quarter
1. **Implement mode registry** - full implementation
2. **Implement provider registry** - full implementation
3. **Consolidate retry logic** - eliminate duplication
4. **Add lazy loading** - performance boost

### Long-term (2025-2026)
1. **Ship v2.0** with breaking changes
2. **Build plugin ecosystem**
3. **Performance optimization**
4. **Third-party provider support**

---

## Conclusion

Instructor is already a powerful library. Version 2.0 will make it:

- **Faster**: Lazy loading, caching, optimized streaming
- **Cleaner**: Registry patterns, eliminated duplication, modern Python
- **More Extensible**: Plugin architecture, provider isolation
- **Better DX**: Rich errors, comprehensive docs, automated migration

The incremental approach means you can start delivering value immediately while working toward the larger vision. Each phase is independently useful and doesn't require the others to be complete.

**The key is to start small**, validate the patterns, and build momentum. Quality of life improvements can be delivered immediately while architectural changes are being designed and implemented.
