# Phase 2: Provider Registry

**Status**: Not Started
**Priority**: P0 (Critical for decoupling)
**Est. Duration**: 4-6 weeks
**Est. Effort**: 20-25 days
**Assignee**: TBD
**Dependencies**: Phase 1 (Mode Registry) helpful but not required

---

## Quick Reference

| Component | Current | Target | Impact |
|-----------|---------|--------|--------|
| auto_client.py | 924 lines | ~100-150 lines | 78-84% reduction |
| Provider branches | 19 if/elif | Registry lookup | Extensible |
| Duplication | 85% across branches | <10% | DRY principle |
| Add new provider | Edit core file | Register independently | Plugin-ready |

---

## Overview

Replace the giant if/elif chain in auto_client.py with a provider registry that enables dynamic provider registration, eliminates duplication, and supports plugin architecture.

### Current Problem

**File**: `instructor/auto_client.py`
**Lines**: 157-1080 (924 lines total)
**Pattern**: 19 provider branches with 85% duplication

**Current Code Structure** (repeated 19 times):

```python
elif provider == "anthropic":
    try:
        # 1. Import (2-4 lines)
        import anthropic
        from instructor import from_anthropic

        # 2. API Key Handling (0-5 lines)
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        # 3. Client Instantiation (4-6 lines)
        client = (
            anthropic.AsyncAnthropic(api_key=api_key)
            if async_client
            else anthropic.Anthropic(api_key=api_key)
        )

        # 4. Factory Call (3-8 lines)
        result = from_anthropic(client, model=model_name, mode=mode, **kwargs)

        # 5. Logging (4 lines)
        logger.info("Client initialized", extra={**provider_info, "status": "success"})

        # 6. Return (1 line)
        return result

    except ImportError:
        raise ConfigurationError("anthropic package required...")
    except Exception as e:
        logger.error("Error initializing...", exc_info=True)
        raise
```

**19 Providers**: openai, azure_openai, anthropic, google, mistral, cohere, perplexity, groq, writer, bedrock, cerebras, fireworks, vertexai (deprecated), generative-ai (deprecated), ollama, deepseek, xai, openrouter, litellm

**Duplication**: ~40 lines × 19 = ~760 lines of nearly identical code

### Goals

1. **Provider Registry**: Central registration system
2. **Factory Pattern**: Providers register factory functions
3. **Metadata**: Provider capabilities, requirements, defaults
4. **Lazy Loading**: Only load provider when used
5. **Extensibility**: Third-party providers without core changes

### Success Metrics

- ☐ auto_client.py reduced from 924 lines to <150 lines
- ☐ All 19 providers migrated to registry
- ☐ No hardcoded provider logic in core
- ☐ Type-safe provider registration
- ☐ All tests passing
- ☐ Plugin API defined

---

## Current State Analysis

### Provider Patterns (from MEASUREMENTS.md)

| Pattern | Count | Examples | Key Characteristics |
|---------|-------|----------|---------------------|
| Standard | 13 (68%) | openai, anthropic, groq, writer | Simple import→instantiate→wrap |
| OpenAI-compatible | 5 (26%) | perplexity, deepseek, openrouter | Use openai package with custom base_url |
| Special | 4 (21%) | bedrock, ollama, litellm | Complex config or unique handling |

### Provider Metadata Needed

Each provider needs:
- **Package name**: For import checking
- **Client classes**: Sync and async
- **Factory function**: `from_provider` reference
- **Default mode**: Preferred mode
- **Supported modes**: List of valid modes
- **Env var**: API key environment variable
- **Base URL**: For OpenAI-compatible providers
- **Config**: Special configuration (AWS creds, model-based modes, etc.)

---

## Proposed Solution

### Architecture

```
┌────────────────────────────────────────────────────────┐
│                  ProviderRegistry                       │
│  - _providers: dict[str, ProviderConfig]               │
│  - _factories: dict[str, ProviderFactory]              │
│                                                         │
│  + register(name, config, factory)                     │
│  + get_factory(name) -> ProviderFactory                │
│  + list_providers() -> list[str]                       │
│  + create_client(name, **kwargs) -> Instructor         │
└────────────────────────────────────────────────────────┘
                           │
                           │ uses
                           ▼
┌────────────────────────────────────────────────────────┐
│                  ProviderConfig                         │
│  - package_name: str                                   │
│  - default_mode: Mode                                  │
│  - supported_modes: list[Mode]                         │
│  - env_var_name: str | None                            │
│  - docs_url: str                                       │
│  - deprecated: bool                                    │
└────────────────────────────────────────────────────────┘
                           │
                           │ used by
                           ▼
┌────────────────────────────────────────────────────────┐
│               ProviderFactory Protocol                  │
│  + create_client(api_key, model, mode, async, **kw)   │
└────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Provider Config

**File**: `instructor/core/provider_config.py` (new)

```python
from dataclasses import dataclass, field
from typing import Optional
from instructor.mode import Mode

@dataclass
class ProviderConfig:
    """
    Configuration metadata for a provider.

    This defines everything the registry needs to know about a provider
    without actually loading the provider code.
    """

    # Required fields
    name: str
    package_name: str  # For import checking
    default_mode: Mode

    # Optional fields
    supported_modes: list[Mode] = field(default_factory=list)
    env_var_name: Optional[str] = None
    env_vars: dict[str, str] = field(default_factory=dict)  # Multiple env vars
    docs_url: str = "https://docs.instructor.com"
    install_cmd: Optional[str] = None
    deprecated: bool = False
    deprecation_message: Optional[str] = None

    # OpenAI-compatible providers
    base_url: Optional[str] = None
    uses_openai_client: bool = False

    # Special configuration
    requires_auth: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.deprecated and not self.deprecation_message:
            self.deprecation_message = (
                f"{self.name} provider is deprecated. "
                f"See {self.docs_url} for migration guide."
            )

        if not self.install_cmd:
            self.install_cmd = f"pip install {self.package_name}"

        # Default docs URL
        if self.docs_url == "https://docs.instructor.com":
            self.docs_url = f"https://docs.instructor.com/integrations/{self.name}"
```

#### 2. Provider Factory Protocol

**File**: `instructor/core/provider_factory.py` (new)

```python
from typing import Protocol, Any, Optional
from instructor.mode import Mode
from instructor.client import Instructor, AsyncInstructor

class ProviderFactory(Protocol):
    """
    Protocol for provider factory functions.

    All providers must implement this interface to be registered.
    """

    def create_client(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        mode: Optional[Mode] = None,
        async_client: bool = False,
        **kwargs: Any
    ) -> Instructor | AsyncInstructor:
        """
        Create an instructor client for this provider.

        Args:
            api_key: API key (or None to use env var)
            model: Model name
            mode: Mode to use (or None for default)
            async_client: If True, return AsyncInstructor
            **kwargs: Provider-specific arguments

        Returns:
            Patched instructor client

        Raises:
            ImportError: If provider package not installed
            ConfigurationError: If API key missing or other config error
        """
        ...
```

#### 3. Provider Registry

**File**: `instructor/core/provider_registry.py` (new)

```python
import importlib
import logging
from typing import Callable, Optional
from instructor.core.provider_config import ProviderConfig
from instructor.core.provider_factory import ProviderFactory
from instructor.core.exceptions import ConfigurationError
from instructor.mode import Mode

logger = logging.getLogger(__name__)

class ProviderRegistry:
    """
    Central registry for provider factories.

    Manages provider registration, validation, and client creation.
    """

    def __init__(self):
        self._configs: dict[str, ProviderConfig] = {}
        self._factories: dict[str, ProviderFactory] = {}
        self._lazy_factories: dict[str, Callable[[], ProviderFactory]] = {}

    def register(
        self,
        config: ProviderConfig,
        factory: Optional[ProviderFactory] = None,
        lazy_factory: Optional[Callable[[], ProviderFactory]] = None,
        replace: bool = False
    ) -> None:
        """
        Register a provider.

        Args:
            config: Provider configuration
            factory: Pre-created factory (eager)
            lazy_factory: Function to create factory (lazy)
            replace: Allow replacing existing provider
        """
        name = config.name

        if name in self._configs and not replace:
            raise ValueError(
                f"Provider '{name}' already registered. "
                f"Use replace=True to override."
            )

        if factory is None and lazy_factory is None:
            raise ValueError("Must provide factory or lazy_factory")

        # Store config
        self._configs[name] = config

        # Store factory
        if factory is not None:
            self._factories[name] = factory
            self._lazy_factories.pop(name, None)
            logger.debug(f"Registered provider '{name}' (eager)")
        else:
            self._lazy_factories[name] = lazy_factory
            logger.debug(f"Registered provider '{name}' (lazy)")

    def get_factory(self, name: str) -> ProviderFactory:
        """Get factory for provider, loading lazily if needed."""
        # Check if loaded
        if name in self._factories:
            return self._factories[name]

        # Check if lazy loader exists
        if name in self._lazy_factories:
            logger.debug(f"Lazy loading factory for '{name}'")
            factory = self._lazy_factories[name]()
            self._factories[name] = factory
            del self._lazy_factories[name]
            return factory

        # Not found
        raise ValueError(
            f"Unknown provider: {name}. "
            f"Available: {', '.join(self.list_providers())}"
        )

    def get_config(self, name: str) -> ProviderConfig:
        """Get configuration for provider."""
        if name not in self._configs:
            raise ValueError(f"Unknown provider: {name}")
        return self._configs[name]

    def create_client(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        mode: Optional[Mode] = None,
        async_client: bool = False,
        **kwargs
    ):
        """
        Create instructor client for provider.

        This is the main entry point that replaces auto_client.py logic.
        """
        # Get config
        config = self.get_config(provider)

        # Check if deprecated
        if config.deprecated:
            import warnings
            warnings.warn(
                config.deprecation_message,
                DeprecationWarning,
                stacklevel=2
            )

        # Check if package installed
        try:
            importlib.import_module(config.package_name)
        except ImportError as e:
            raise ConfigurationError(
                f"The {config.package_name} package is required for {provider}. "
                f"Install with: {config.install_cmd}\n"
                f"See: {config.docs_url}"
            ) from e

        # Get API key from env if not provided
        if api_key is None and config.env_var_name:
            import os
            api_key = os.environ.get(config.env_var_name)

        # Validate API key if required
        if config.requires_auth and not api_key:
            raise ConfigurationError(
                f"API key required for {provider}. "
                f"Set {config.env_var_name} environment variable or pass api_key parameter. "
                f"See: {config.docs_url}"
            )

        # Use default mode if not specified
        if mode is None:
            mode = config.default_mode

        # Validate mode is supported
        if config.supported_modes and mode not in config.supported_modes:
            raise ValueError(
                f"Mode {mode} not supported by {provider}. "
                f"Supported modes: {', '.join(str(m) for m in config.supported_modes)}"
            )

        # Get factory and create client
        try:
            factory = self.get_factory(provider)
            result = factory.create_client(
                api_key=api_key,
                model=model,
                mode=mode,
                async_client=async_client,
                **kwargs
            )

            logger.info(
                f"Created {provider} client",
                extra={"provider": provider, "mode": str(mode), "async": async_client}
            )

            return result

        except Exception as e:
            logger.error(
                f"Error creating {provider} client: {e}",
                exc_info=True,
                extra={"provider": provider, "error": str(e)}
            )
            raise

    def list_providers(self) -> list[str]:
        """List all registered providers."""
        return sorted(self._configs.keys())

    def has_provider(self, name: str) -> bool:
        """Check if provider is registered."""
        return name in self._configs

    def stats(self) -> dict:
        """Get registry statistics."""
        return {
            "total": len(self._configs),
            "loaded": len(self._factories),
            "lazy": len(self._lazy_factories),
            "deprecated": sum(1 for c in self._configs.values() if c.deprecated)
        }

# Global registry
provider_registry = ProviderRegistry()

# Decorator for registration
def register_provider(config: ProviderConfig, lazy: bool = True, replace: bool = False):
    """
    Decorator to register a provider factory.

    Example:
        @register_provider(ANTHROPIC_CONFIG, lazy=True)
        def create_anthropic_client(...):
            ...
    """
    def decorator(factory_func):
        if lazy:
            provider_registry.register(
                config,
                lazy_factory=lambda: factory_func,
                replace=replace
            )
        else:
            provider_registry.register(
                config,
                factory=factory_func,
                replace=replace
            )
        return factory_func
    return decorator
```

---

## Implementation Steps

### Step 1: Create Core Infrastructure (Week 1)

Create 3 new files:
1. `instructor/core/provider_config.py`
2. `instructor/core/provider_factory.py`
3. `instructor/core/provider_registry.py`

**Test**:
```python
# tests/core/test_provider_registry.py
from instructor.core.provider_registry import ProviderRegistry, register_provider
from instructor.core.provider_config import ProviderConfig
from instructor.mode import Mode

def test_register_provider():
    registry = ProviderRegistry()

    config = ProviderConfig(
        name="test",
        package_name="test_package",
        default_mode=Mode.TOOLS,
        supported_modes=[Mode.TOOLS, Mode.JSON]
    )

    def test_factory(...):
        return Mock()

    registry.register(config, factory=test_factory)

    assert registry.has_provider("test")
    assert registry.get_factory("test") is test_factory
```

### Step 2: Migrate One Provider (Week 2)

**Choose Anthropic** (straightforward, no special handling)

**Create**: `instructor/providers/anthropic/factory.py`

```python
from instructor.core.provider_registry import register_provider, ProviderConfig
from instructor.core.provider_factory import ProviderFactory
from instructor.mode import Mode
from typing import Optional, Any

# Define configuration
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
    docs_url="https://docs.instructor.com/integrations/anthropic",
    supports_vision=True,
    supports_streaming=True,
)

@register_provider(ANTHROPIC_CONFIG, lazy=True)
def create_anthropic_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    mode: Optional[Mode] = None,
    async_client: bool = False,
    **kwargs: Any
):
    """Factory for Anthropic instructor clients."""
    import anthropic
    from instructor import from_anthropic

    # Create native client
    client = (
        anthropic.AsyncAnthropic(api_key=api_key)
        if async_client
        else anthropic.Anthropic(api_key=api_key)
    )

    # Wrap with instructor
    return from_anthropic(
        client,
        model=model,
        mode=mode or ANTHROPIC_CONFIG.default_mode,
        **kwargs
    )
```

**Import in `__init__.py`**:
```python
# instructor/providers/anthropic/__init__.py
from . import factory  # Triggers registration
```

### Step 3: Update auto_client.py (Week 2-3)

**Before** (924 lines):
```python
def from_provider(provider, ...):
    if provider == "openai":
        # 40 lines
    elif provider == "anthropic":
        # 40 lines
    # ... 17 more
```

**After** (~100 lines):
```python
from instructor.core.provider_registry import provider_registry

def from_provider(
    model: str,
    api_key: Optional[str] = None,
    async_client: bool = False,
    cache: Optional[Any] = None,
    mode: Optional[Mode] = None,
    **kwargs
):
    """
    Create instructor client from provider/model string.

    Args:
        model: "provider/model-name" format
        api_key: API key (or use env var)
        async_client: Return AsyncInstructor if True
        cache: Cache configuration
        mode: Mode to use
        **kwargs: Provider-specific args

    Returns:
        Instructor or AsyncInstructor client
    """
    # Parse provider from model
    if "/" not in model:
        raise ValueError(
            f"Model must be in 'provider/model-name' format. Got: {model}"
        )

    provider, model_name = model.split("/", 1)
    provider = provider.lower()

    # Use registry to create client
    return provider_registry.create_client(
        provider=provider,
        api_key=api_key,
        model=model_name,
        mode=mode,
        async_client=async_client,
        **kwargs
    )
```

### Step 4: Migrate Remaining Providers (Weeks 3-5)

**Standard providers** (1-2 days each):
- OpenAI, Cohere, Groq, Writer, Cerebras, Fireworks, xAI

**OpenAI-compatible** (similar pattern):
- Perplexity, DeepSeek, OpenRouter

**Special cases** (need custom logic):
- Bedrock (AWS credentials, model-based mode)
- Ollama (model-based mode selection)
- LiteLLM (function-based, not client)

**Example for Bedrock**:
```python
# instructor/providers/bedrock/factory.py

BEDROCK_CONFIG = ProviderConfig(
    name="bedrock",
    package_name="boto3",
    default_mode=Mode.BEDROCK_TOOLS,
    supported_modes=[Mode.BEDROCK_TOOLS, Mode.BEDROCK_JSON],
    env_vars={
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "region": "AWS_DEFAULT_REGION",
    },
    docs_url="https://docs.instructor.com/integrations/bedrock",
    requires_auth=False,  # Uses AWS credentials, not API key
)

@register_provider(BEDROCK_CONFIG, lazy=True)
def create_bedrock_client(...):
    import boto3
    from instructor import from_bedrock

    # Extract AWS credentials
    aws_kwargs = {}
    for key, env_var in BEDROCK_CONFIG.env_vars.items():
        if key in kwargs:
            aws_kwargs[key] = kwargs.pop(key)
        elif env_var in os.environ:
            aws_kwargs[key] = os.environ[env_var]

    # Create boto3 client
    client = boto3.client("bedrock-runtime", **aws_kwargs)

    # Auto-detect mode based on model
    if mode is None:
        if model and ("claude" in model.lower() or "anthropic" in model.lower()):
            mode = Mode.BEDROCK_TOOLS
        else:
            mode = Mode.BEDROCK_JSON

    return from_bedrock(client, model=model, mode=mode, **kwargs)
```

### Step 5: Remove Old Code (Week 6)

**Delete from auto_client.py**:
- Lines 157-1080 (the giant if/elif chain)
- Keep only the new `from_provider()` function

**Before**: 924 lines
**After**: ~100 lines
**Reduction**: 89%

### Step 6: Documentation (Week 6)

**Create**: `docs/concepts/provider_registry.md`

```markdown
# Provider Registry

The Provider Registry enables dynamic provider management in Instructor.

## For Users

Use the `from_provider()` function:

```python
import instructor

# Auto-detects provider from model string
client = instructor.from_provider(
    model="anthropic/claude-3-5-sonnet-20241022",
    api_key="your-key"
)
```

## For Provider Developers

### Adding a New Provider

1. Create provider config:

```python
from instructor.core.provider_config import ProviderConfig
from instructor.mode import Mode

YOUR_PROVIDER_CONFIG = ProviderConfig(
    name="your_provider",
    package_name="your_provider_sdk",
    default_mode=Mode.YOUR_PROVIDER_TOOLS,
    supported_modes=[Mode.YOUR_PROVIDER_TOOLS, Mode.YOUR_PROVIDER_JSON],
    env_var_name="YOUR_PROVIDER_API_KEY",
)
```

2. Create factory function:

```python
from instructor.core.provider_registry import register_provider

@register_provider(YOUR_PROVIDER_CONFIG, lazy=True)
def create_your_provider_client(...):
    import your_provider_sdk
    from instructor import from_your_provider

    client = your_provider_sdk.Client(api_key=api_key)
    return from_your_provider(client, ...)
```

3. Import in `__init__.py`:

```python
from . import factory  # Triggers registration
```

That's it! No core file modifications needed.
```

---

## Testing Strategy

### Unit Tests

```python
# tests/core/test_provider_registry.py

def test_provider_registration():
    """Test basic registration."""
    ...

def test_lazy_loading():
    """Test providers load on first use."""
    ...

def test_duplicate_registration_error():
    """Test error when registering same provider twice."""
    ...

def test_unknown_provider_error():
    """Test helpful error for unknown provider."""
    ...

def test_mode_validation():
    """Test mode validation against supported modes."""
    ...
```

### Integration Tests

```bash
# Test with real providers
uv run pytest tests/llm/test_anthropic/ -v
uv run pytest tests/llm/test_openai/ -v

# Test auto_client with registry
uv run pytest tests/test_auto_client.py -v
```

---

## Success Criteria

Phase 2 is **complete** when:

- ☐ ProviderRegistry implemented
- ☐ All 19 providers migrated to registry
- ☐ auto_client.py reduced to <150 lines
- ☐ All provider tests passing
- ☐ No hardcoded provider logic in core
- ☐ Documentation complete
- ☐ Plugin API defined

---

## Rollback Plan

**Feature flag during migration**:
```python
USE_PROVIDER_REGISTRY = os.getenv("INSTRUCTOR_USE_PROVIDER_REGISTRY", "false") == "true"

if USE_PROVIDER_REGISTRY:
    return provider_registry.create_client(...)
else:
    # Old if/elif chain
    ...
```

---

## Next Phase

- **[Phase 3: Lazy Loading](./phase3_lazy_loading.md)** - Optimize import time
- **[Phase 6: Auto Client Refactor](./phase6_auto_client_refactor.md)** - Finalize auto_client.py

---

**Status**: Ready after Phase 1
**Timeline**: 4-6 weeks
