# Phase 6: Auto Client Refactor

**Status**: Not Started
**Priority**: P1 (Finalize architecture)
**Est. Duration**: 2-3 weeks
**Est. Effort**: 8-10 days
**Dependencies**: Phases 1, 2, 3 must be complete

---

## Quick Reference

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| auto_client.py | 924 lines | <100 lines | 89% reduction |
| Logic | 19 if/elif branches | Registry lookup | Clean |
| Maintainability | Edit core for new providers | Provider self-registers | Plugin-ready |

---

## Overview

Final cleanup of `auto_client.py` after registries are in place. Remove all hardcoded provider logic and use pure registry-based dispatch.

### Current State (After Phases 1-3)

Phases 1-3 create the infrastructure:
- **Phase 1**: Mode registry with lazy loading
- **Phase 2**: Provider registry with lazy loading
- **Phase 3**: Entry points for zero-load imports

But `auto_client.py` still has the old 924-line if/elif chain with a feature flag.

### Goal

**Final `auto_client.py`** (< 100 lines total):

```python
"""Auto client creation from model strings."""

from typing import Optional, Any
from instructor.mode import Mode
from instructor.core.provider_registry import provider_registry

def from_provider(
    model: str,
    api_key: Optional[str] = None,
    async_client: bool = False,
    mode: Optional[Mode] = None,
    **kwargs: Any
):
    """
    Create instructor client from model string.

    Args:
        model: Format "provider/model-name" (e.g., "anthropic/claude-3-5-sonnet")
        api_key: API key (or None to use environment variable)
        async_client: Return AsyncInstructor if True
        mode: Mode to use (or None for provider default)
        **kwargs: Provider-specific arguments

    Returns:
        Instructor or AsyncInstructor client

    Examples:
        >>> client = from_provider("openai/gpt-4")
        >>> client = from_provider("anthropic/claude-3-5-sonnet", mode=Mode.ANTHROPIC_TOOLS)
        >>> async_client = from_provider("gemini/gemini-1.5-pro", async_client=True)
    """
    # Parse provider from model
    if "/" not in model:
        raise ValueError(
            f"Model must be in 'provider/model-name' format. "
            f"Examples: 'openai/gpt-4', 'anthropic/claude-3-5-sonnet'. "
            f"Got: {model}"
        )

    provider, model_name = model.split("/", 1)
    provider = provider.lower()

    # Delegate to provider registry
    return provider_registry.create_client(
        provider=provider,
        api_key=api_key,
        model=model_name,
        mode=mode,
        async_client=async_client,
        **kwargs
    )

def get_provider(model: str) -> str:
    """
    Extract provider from model string.

    Args:
        model: Format "provider/model-name"

    Returns:
        Provider name

    Examples:
        >>> get_provider("openai/gpt-4")
        'openai'
    """
    if "/" not in model:
        raise ValueError(f"Model must be in 'provider/model-name' format. Got: {model}")

    provider, _ = model.split("/", 1)
    return provider.lower()

# Backward compatibility: supported_providers
@property
def supported_providers() -> list[str]:
    """
    List all supported providers.

    Returns:
        List of provider names
    """
    return provider_registry.list_providers()
```

**That's it!** ~80 lines vs 924 lines.

---

## Implementation Steps

### Step 1: Remove Feature Flag (Day 1)

**Current** (transitional code):

```python
USE_PROVIDER_REGISTRY = os.getenv("INSTRUCTOR_USE_PROVIDER_REGISTRY", "false") == "true"

if USE_PROVIDER_REGISTRY:
    # New registry code
    return provider_registry.create_client(...)
else:
    # Old 900-line if/elif chain
    ...
```

**New**: Remove the flag, use registry only

```python
# Just use the registry
return provider_registry.create_client(...)
```

### Step 2: Delete Old Code (Day 1)

Delete lines 157-1080 (the entire if/elif chain).

Keep only:
- Imports
- `from_provider()` function
- `get_provider()` helper
- `supported_providers` property (now using registry)

### Step 3: Improve Error Messages (Day 2)

Add helpful errors with registry info:

```python
def from_provider(model: str, ...):
    """..."""
    # Validate format
    if "/" not in model:
        available = provider_registry.list_providers()
        raise ValueError(
            f"Model must be in 'provider/model-name' format. "
            f"Available providers: {', '.join(available[:5])}... "
            f"Example: '{available[0]}/model-name'. "
            f"Got: {model}"
        )

    provider, model_name = model.split("/", 1)
    provider = provider.lower()

    # Check if provider exists
    if not provider_registry.has_provider(provider):
        available = provider_registry.list_providers()
        similar = _find_similar(provider, available)  # Fuzzy match

        error_msg = f"Unknown provider: {provider}. "
        if similar:
            error_msg += f"Did you mean '{similar[0]}'? "
        error_msg += f"Available providers: {', '.join(available)}"

        raise ValueError(error_msg)

    # Delegate to registry
    try:
        return provider_registry.create_client(...)
    except ImportError as e:
        # Add helpful context
        config = provider_registry.get_config(provider)
        raise ImportError(
            f"{str(e)}\n\n"
            f"Install with: {config.install_cmd}\n"
            f"Documentation: {config.docs_url}"
        ) from e
```

### Step 4: Add Convenience Functions (Day 3)

**List providers with details**:

```python
def list_providers(verbose: bool = False) -> list[str] | dict:
    """
    List all available providers.

    Args:
        verbose: If True, return dict with provider details

    Returns:
        List of provider names, or dict with details if verbose=True
    """
    if not verbose:
        return provider_registry.list_providers()

    # Verbose: return details
    result = {}
    for name in provider_registry.list_providers():
        config = provider_registry.get_config(name)
        result[name] = {
            'default_mode': str(config.default_mode),
            'supported_modes': [str(m) for m in config.supported_modes],
            'package': config.package_name,
            'docs': config.docs_url,
            'deprecated': config.deprecated,
        }
    return result

# Usage:
# >>> list_providers()
# ['anthropic', 'openai', 'gemini', ...]
#
# >>> list_providers(verbose=True)
# {
#   'anthropic': {
#     'default_mode': 'anthropic_tools',
#     'supported_modes': ['anthropic_tools', 'anthropic_json', ...],
#     'package': 'anthropic',
#     'docs': 'https://...',
#     'deprecated': False
#   },
#   ...
# }
```

**Check provider availability**:

```python
def is_provider_available(provider: str) -> bool:
    """
    Check if provider is installed and available.

    Args:
        provider: Provider name

    Returns:
        True if provider package is installed
    """
    if not provider_registry.has_provider(provider):
        return False

    config = provider_registry.get_config(provider)

    try:
        __import__(config.package_name)
        return True
    except ImportError:
        return False

# Usage:
# >>> is_provider_available('anthropic')
# True
# >>> is_provider_available('notinstalled')
# False
```

### Step 5: Update Tests (Days 4-5)

**Update existing tests**:

```python
# tests/test_auto_client.py

def test_from_provider_openai():
    """Test from_provider with OpenAI."""
    client = from_provider(
        "openai/gpt-4",
        api_key="test-key"
    )
    assert isinstance(client, Instructor)

def test_from_provider_invalid_format():
    """Test error for invalid model format."""
    with pytest.raises(ValueError, match="provider/model-name"):
        from_provider("gpt-4")  # Missing provider prefix

def test_from_provider_unknown_provider():
    """Test error for unknown provider."""
    with pytest.raises(ValueError, match="Unknown provider"):
        from_provider("unknown/model")

def test_from_provider_missing_package():
    """Test error when provider package not installed."""
    with pytest.raises(ImportError, match="Install with"):
        from_provider("some-provider/model")

def test_list_providers():
    """Test listing providers."""
    providers = list_providers()
    assert 'openai' in providers
    assert 'anthropic' in providers

def test_list_providers_verbose():
    """Test verbose provider listing."""
    providers = list_providers(verbose=True)
    assert 'openai' in providers
    assert 'default_mode' in providers['openai']
    assert 'supported_modes' in providers['openai']
```

**Run full test suite**:

```bash
# Should all pass with new implementation
uv run pytest tests/test_auto_client.py -v
uv run pytest tests/ -k "not llm" -v  # All unit tests
```

### Step 6: Update Documentation (Days 6-7)

**Update**:
- `docs/integrations/auto_client.md`
- `README.md` examples
- Migration guide

**Document**:
```markdown
## Auto Client (from_provider)

The `from_provider()` function automatically creates the correct instructor client based on the model string.

### Basic Usage

```python
import instructor

# OpenAI
client = instructor.from_provider("openai/gpt-4")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# With async
async_client = instructor.from_provider(
    "anthropic/claude-3-5-sonnet",
    async_client=True
)

# With specific mode
client = instructor.from_provider(
    "anthropic/claude-3-5-sonnet",
    mode=instructor.Mode.ANTHROPIC_TOOLS
)
```

### Listing Providers

```python
# Simple list
providers = instructor.list_providers()
# ['anthropic', 'openai', 'gemini', ...]

# Detailed info
providers = instructor.list_providers(verbose=True)
# {'anthropic': {'default_mode': '...', 'supported_modes': [...], ...}, ...}

# Check if provider available
if instructor.is_provider_available('anthropic'):
    client = instructor.from_provider("anthropic/...")
```

### Error Handling

```python
try:
    client = instructor.from_provider("typo/model")
except ValueError as e:
    # "Unknown provider: typo. Did you mean 'openai'? Available: ..."
    print(e)

try:
    client = instructor.from_provider("provider-not-installed/model")
except ImportError as e:
    # "Package required... Install with: pip install ..."
    print(e)
```
```

---

## Testing Strategy

### Regression Tests

Ensure all existing functionality works:

```bash
# All auto_client tests should pass
uv run pytest tests/test_auto_client.py -v

# All provider tests should work via auto_client
uv run pytest tests/llm/ -v
```

### Performance Tests

Verify import time improvements:

```python
import time
start = time.time()
from instructor import from_provider
elapsed = time.time() - start

assert elapsed < 0.01  # Should be instant (lazy loading)
```

---

## Success Criteria

Phase 6 is **complete** when:

- ☐ Feature flag removed
- ☐ Old if/elif chain deleted (924 lines → <100 lines)
- ☐ Improved error messages with suggestions
- ☐ Convenience functions added (list_providers, is_provider_available)
- ☐ All tests passing
- ☐ Documentation updated
- ☐ No regressions in functionality

---

## Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Lines of code | 1,080 | <100 | 91% reduction |
| Cyclomatic complexity | Very high (19 branches) | Low (registry lookup) | Much simpler |
| Maintainability | Edit core for new providers | Providers self-register | Plugin-ready |
| Error messages | Generic | Helpful with suggestions | Better DX |

---

## Migration Notes

**No breaking changes**:
- `from_provider()` API unchanged
- All existing code continues to work
- Just using registries under the hood

**New features**:
- `list_providers(verbose=True)` for provider discovery
- `is_provider_available()` for checking installed providers
- Better error messages

---

## Next Steps

After Phase 6:
- Architecture refactoring (Theme 2) is **complete**
- Ready for Theme 3 (Performance optimization)
- Ready for Theme 4 (Developer experience)
- v2.0 phases (4, 7) can proceed

---

**Status**: Ready after Phases 1-3
**Timeline**: 2-3 weeks
