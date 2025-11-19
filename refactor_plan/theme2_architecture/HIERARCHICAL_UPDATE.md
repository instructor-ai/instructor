# Mode Design Updates

**Date**: 2025-11-06
**Status**: Design updated
**Impact**: Phase 1 (Mode Registry) uses flat Mode enum

---

## Summary

Phase 1 uses **flat Mode enum** with `(Provider, Mode)` tuples in the registry. The number of modes will be reduced over time through consolidation.

### Key Changes

#### 1. Mode Definition

**Before** (Flat):
```python
class Mode(Enum):
    GEMINI_JSON = "gemini_json"
    GEMINI_TOOLS = "gemini_tools"
    OPENAI_JSON = "json_mode"
    # ... 39 more flat modes
```

**After** (Registry with Mode enum):
```python
class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    # ... ~15 providers

class Mode(Enum):
    # Keep flat Mode enum, reduce modes over time
    GEMINI_JSON = "gemini_json"
    GEMINI_TOOLS = "gemini_tools"
    ANTHROPIC_TOOLS = "anthropic_tools"
    # ... fewer modes as we consolidate

# Registry uses (Provider, Mode) tuples
type ModeKey = tuple[Provider, Mode]
```

#### 2. Registry Signature

**Before**:
```python
mode_registry.register(
    mode=Mode.GEMINI_JSON,
    handler=GeminiJSONHandler()
)
```

**After**:
```python
mode_registry.register(
    provider=Provider.GEMINI,
    mode=Mode.GEMINI_JSON,
    handler=GeminiJSONHandler()
)
```

#### 3. Registration Decorator

**Before**:
```python
@register_mode_handler(Mode.ANTHROPIC_TOOLS, lazy=True)
class AnthropicToolsHandler:
    ...
```

**After**:
```python
@register_mode_handler(Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS, lazy=True)
class AnthropicToolsHandler:
    ...
```

#### 4. New Query Methods

```python
# Get all modes for a provider
gemini_modes = mode_registry.get_modes_for_provider(Provider.GEMINI)
# → [Mode.GEMINI_JSON, Mode.GEMINI_TOOLS]

# Get all providers supporting a mode
json_providers = mode_registry.get_providers_for_mode(Mode.GEMINI_JSON)
# → [Provider.GEMINI]
```

---

## Benefits

1. **Simpler design**: Keep flat Mode enum, reduce modes over time
2. **Backward compatible**: Existing Mode enum values continue to work
3. **Queryable**: Can filter by provider or mode pattern
4. **Clear semantics**: `(Provider.GEMINI, Mode.GEMINI_JSON)` is explicit
5. **Easier to extend**: Add new modes as needed, consolidate similar ones over time

---

## Implementation Status

### Updated Documents

- ✅ `phase1_mode_registry.md` - Added hierarchical design section
- ⏳ Full registry implementation code needs update (postponed - can be done during implementation)
- ⏳ Phase 4 needs update (now just adds metadata, not hierarchical structure)

### What's Left

The **concept and API** are documented in Phase 1. The **detailed code implementation** in Phase 1 still shows flat Mode in some places but can be updated during actual implementation.

**Recommendation**: Use hierarchical design from the start when implementing Phase 1.

---

## Backward Compatibility

Support old flat Mode enum via aliases:

```python
# instructor/mode.py
class Mode(Enum):
    """Mode enum - will be reduced over time through consolidation."""
    GEMINI_JSON = "gemini_json"
    GEMINI_TOOLS = "gemini_tools"
    # ... modes will be consolidated over time

# Registry uses (Provider, Mode) tuples
mode_registry.get_handler(Provider.GEMINI, Mode.GEMINI_JSON)
```

---

## Phase 4 Impact

**Phase 4 (Mode Metadata)** adds metadata to modes:

**Before**: Introduce hierarchical structure (not needed)
**After**: Add rich metadata to flat Mode enum

Phase 4 now adds:
- ModeCapability enum
- ModeDescriptor dataclass with metadata
- Capability queries (supports_vision, supports_streaming, etc.)
- Not the hierarchical structure itself (that's now in Phase 1)

---

## Next Steps

1. ✅ Phase 1 document updated to use flat Mode enum
2. ⏳ Update Phase 4 to reflect metadata-only role
3. ⏳ During Phase 1 implementation, use (Provider, Mode) tuples in registry
4. ⏳ Consolidate similar modes over time to reduce total count

---

**Status**: Architectural decision made, documentation updated
**Implementation**: Use flat Mode enum with (Provider, Mode) tuples in registry
