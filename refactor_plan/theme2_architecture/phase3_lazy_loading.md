# Phase 3: Lazy Provider Loading

**Status**: Not Started
**Priority**: P1 (Performance)
**Est. Duration**: 2-3 weeks
**Est. Effort**: 10-12 days
**Dependencies**: Phases 1 & 2 (Registries) must be complete

---

## Quick Reference

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Import time | ~500ms | <100ms | 80% faster |
| Memory (initial) | All providers loaded | Core only | ~60% reduction |
| Providers loaded | 11 (all) | 0 (on demand) | Lazy |

---

## Overview

Implement lazy loading throughout the stack to drastically reduce import time and memory footprint. Only load providers and mode handlers when actually used.

### Current Problem

```python
# When you do this:
import instructor

# This happens (in order):
# 1. Load all 11 provider utils (3,488 lines)
# 2. Import all handler functions
# 3. Build dispatch dictionaries
# 4. Total: ~500ms, ~20MB memory
```

**Even if you only use OpenAI!**

### Goal

```python
# When you do this:
import instructor

# This happens:
# 1. Load core modules only
# 2. Register provider/mode entry points (metadata only)
# 3. Total: <100ms, ~5MB memory

# Then when you do this:
client = instructor.from_anthropic(...)

# This happens (first time only):
# 1. Import anthropic utils
# 2. Load Anthropic mode handlers
# 3. Create client
```

---

## Implementation Strategy

### Level 1: Provider Lazy Loading (Week 1)

**Current**: `instructor/__init__.py`

```python
# All providers imported eagerly
from .client_anthropic import from_anthropic
from .client_openai import from_openai
# ... 11 more imports
```

**New**:

```python
# instructor/__init__.py
from typing import TYPE_CHECKING

# Only import core
from .client import Instructor, AsyncInstructor
from .mode import Mode

# Type-only imports (no runtime cost)
if TYPE_CHECKING:
    from .client_anthropic import from_anthropic
    from .client_openai import from_openai

# Lazy loading via __getattr__
def __getattr__(name: str):
    """Lazy load provider functions."""
    provider_map = {
        'from_anthropic': ('client_anthropic', 'from_anthropic'),
        'from_openai': ('client_openai', 'from_openai'),
        'from_gemini': ('client_gemini', 'from_gemini'),
        # ... map all providers
    }

    if name in provider_map:
        module_name, attr_name = provider_map[name]
        module = __import__(f'instructor.{module_name}', fromlist=[attr_name])
        return getattr(module, attr_name)

    raise AttributeError(f"module 'instructor' has no attribute '{name}'")
```

**Test**:
```python
import time
start = time.time()
import instructor  # Should be <100ms
elapsed = time.time() - start
assert elapsed < 0.1

# Provider loads on first use
client = instructor.from_anthropic(...)  # Loads anthropic now
```

### Level 2: Mode Handler Lazy Loading (Week 1-2)

Already implemented in Phase 1 via `lazy_loader` parameter:

```python
# Handlers registered with lazy loaders
@register_mode_handler(Mode.ANTHROPIC_TOOLS, lazy=True)
class AnthropicToolsHandler:
    ...
```

**Verify lazy behavior**:
```python
from instructor.core.mode_registry import mode_registry

# Check handler not loaded yet
stats = mode_registry.stats()
assert stats['lazy'] > 0
assert stats['loaded'] == 0

# Use mode - triggers load
handler = mode_registry.get_handler(Mode.ANTHROPIC_TOOLS)

# Now loaded
stats = mode_registry.stats()
assert stats['loaded'] == 1
```

### Level 3: Provider Registry Lazy Loading (Week 2)

Already implemented in Phase 2 via `lazy_factory`:

```python
@register_provider(ANTHROPIC_CONFIG, lazy=True)
def create_anthropic_client(...):
    ...
```

**Optimization**: Import provider modules only when factory is called:

```python
@register_provider(ANTHROPIC_CONFIG, lazy=True)
def create_anthropic_client(...):
    # Import happens HERE, not at registration time
    import anthropic  # <-- Lazy!
    from instructor import from_anthropic  # <-- Lazy!

    client = anthropic.Anthropic(api_key=api_key)
    return from_anthropic(client, ...)
```

### Level 4: Provider Auto-Registration (Week 2-3)

**Current**: Each provider imports factory in `__init__.py`

```python
# instructor/providers/anthropic/__init__.py
from . import factory  # Imports and registers
```

**Problem**: Still loads all provider factory files on import

**Solution**: Entry points for true lazy loading

**File**: `pyproject.toml`

```toml
[project.entry-points."instructor.providers"]
anthropic = "instructor.providers.anthropic.factory"
openai = "instructor.providers.openai.factory"
gemini = "instructor.providers.gemini.factory"
# ... all providers
```

**New registry code**:

```python
# instructor/core/provider_registry.py

def __init__(self):
    self._configs: dict[str, ProviderConfig] = {}
    self._factories: dict[str, ProviderFactory] = {}
    self._entry_points: dict[str, str] = {}  # <-- New
    self._discover_entry_points()

def _discover_entry_points(self):
    """Discover providers via entry points."""
    import importlib.metadata

    for ep in importlib.metadata.entry_points(group="instructor.providers"):
        self._entry_points[ep.name] = ep.value
        logger.debug(f"Discovered provider entry point: {ep.name}")

def get_factory(self, name: str) -> ProviderFactory:
    """Get factory, loading from entry point if needed."""
    if name in self._factories:
        return self._factories[name]

    # Load from entry point
    if name in self._entry_points:
        logger.debug(f"Loading provider '{name}' from entry point")
        module_path = self._entry_points[name]
        module = importlib.import_module(module_path)

        # Factory should have registered itself
        if name not in self._factories:
            raise RuntimeError(
                f"Provider module '{module_path}' did not register provider '{name}'"
            )

        return self._factories[name]

    raise ValueError(f"Unknown provider: {name}")
```

**Result**: Zero providers loaded on `import instructor`

### Level 5: Verify Full Lazy Loading (Week 3)

**Benchmark script**:

```python
import sys
import time

# Measure import time
start = time.time()
import instructor
import_time = time.time() - start

# Check no providers loaded
from instructor.core.provider_registry import provider_registry
from instructor.core.mode_registry import mode_registry

provider_stats = provider_registry.stats()
mode_stats = mode_registry.stats()

print(f"Import time: {import_time*1000:.1f}ms")
print(f"Providers loaded: {provider_stats['loaded']}/{provider_stats['total']}")
print(f"Modes loaded: {mode_stats['loaded']}/{mode_stats['total']}")

# Should be:
# Import time: <100ms
# Providers loaded: 0/19
# Modes loaded: 0/37

assert import_time < 0.1
assert provider_stats['loaded'] == 0
assert mode_stats['loaded'] == 0

# Now use a provider
client = instructor.from_anthropic(
    anthropic.Anthropic(api_key="test"),
    mode=instructor.Mode.ANTHROPIC_TOOLS
)

# Check lazy loading happened
provider_stats = provider_registry.stats()
mode_stats = mode_registry.stats()

print(f"After using Anthropic:")
print(f"Providers loaded: {provider_stats['loaded']}/{provider_stats['total']}")
print(f"Modes loaded: {mode_stats['loaded']}/{mode_stats['total']}")

# Should show Anthropic loaded, others still lazy
```

---

## Additional Optimizations

### Optimize Core Imports

**File**: `instructor/__init__.py`

```python
# Before
from .client import Instructor, AsyncInstructor  # OK
from .mode import Mode  # OK
from .process_response import process_response  # Heavy! Avoid.
from .function_calls import convert_to_openai_tool  # Heavy! Avoid.

# After - only export core classes
__all__ = ['Instructor', 'AsyncInstructor', 'Mode', 'from_provider']

# Heavy modules loaded only when needed
def __getattr__(name: str):
    if name == 'process_response':
        from .process_response import process_response
        return process_response
    # ... etc
```

### Defer Pydantic Imports

```python
# Instead of:
from pydantic import BaseModel  # At module level

# Use:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydantic import BaseModel  # Type-only

def some_function():
    from pydantic import BaseModel  # Import when needed
    ...
```

---

## Testing Strategy

### Import Time Benchmark

```bash
# Create benchmark
cat > benchmark_import.py <<EOF
import time
start = time.time()
import instructor
elapsed = time.time() - start
print(f"{elapsed*1000:.1f}ms")
EOF

# Run multiple times
for i in {1..10}; do
    python benchmark_import.py
done

# Should average <100ms
```

### Memory Benchmark

```python
import sys
import tracemalloc

tracemalloc.start()
import instructor
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Memory: {current / 1024 / 1024:.1f}MB")
# Should be <10MB
```

### Lazy Loading Tests

```python
def test_providers_not_loaded_on_import():
    """Verify providers load lazily."""
    import instructor
    from instructor.core.provider_registry import provider_registry

    stats = provider_registry.stats()
    assert stats['loaded'] == 0, "Providers should not load on import"

def test_provider_loads_on_first_use():
    """Verify provider loads when used."""
    import instructor
    from instructor.core.provider_registry import provider_registry

    # Use provider
    from instructor import from_openai
    import openai
    client = from_openai(openai.OpenAI(api_key="test"))

    stats = provider_registry.stats()
    assert stats['loaded'] > 0, "Provider should load on first use"
```

---

## Success Criteria

Phase 3 is **complete** when:

- ☐ Import time <100ms (measured)
- ☐ Zero providers loaded on import
- ☐ Zero modes loaded on import
- ☐ Lazy loading works for all providers
- ☐ Entry points configured
- ☐ All tests passing
- ☐ Benchmarks documented

---

## Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Import time | ~500ms | <100ms | 80% faster |
| Initial memory | ~20MB | <10MB | 50% less |
| Time to first use | ~500ms + call | ~call time | Same |
| Providers eager | 11 | 0 | 100% lazy |

---

## Next Phase

- **[Phase 4: Hierarchical Modes](./phase4_hierarchical_modes.md)** (v2.0) - Rich mode metadata
- **[Phase 5: Provider Base Migration](./phase5_provider_base_refactor.md)** - Consolidate providers

---

**Status**: Ready after Phases 1 & 2
**Timeline**: 2-3 weeks
