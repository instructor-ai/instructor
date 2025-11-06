# Hierarchical Mode Design Updates

**Date**: 2025-11-06
**Status**: Design approved, implementation pending
**Impact**: Phase 1 (Mode Registry) updated to use hierarchical design

---

## Summary

Phase 1 has been updated to use **hierarchical (Provider, ModeType) design** instead of flat Mode enum.

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

**After** (Hierarchical):
```python
class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    # ... ~15 providers

class ModeType(Enum):
    TOOLS = "tools"
    JSON = "json"
    PARALLEL_TOOLS = "parallel"
    STRUCTURED_OUTPUTS = "structured"
    REASONING_TOOLS = "reasoning"
    # ... ~6 mode types

# Composite mode
type Mode = tuple[Provider, ModeType]
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
    mode_type=ModeType.JSON,
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
@register_mode_handler(Provider.ANTHROPIC, ModeType.TOOLS, lazy=True)
class AnthropicToolsHandler:
    ...
```

#### 4. New Query Methods

```python
# Get all mode types for a provider
gemini_modes = mode_registry.get_modes_for_provider(Provider.GEMINI)
# → [ModeType.JSON, ModeType.TOOLS]

# Get all providers supporting a mode type
json_providers = mode_registry.get_providers_for_mode(ModeType.JSON)
# → [Provider.GEMINI, Provider.OPENAI, Provider.ANTHROPIC, ...]
```

---

## Benefits

1. **Composability**: 15 providers × 6 mode types = any combination vs 42 hardcoded
2. **Fewer enums**: 21 total (15 + 6) vs 42 flat modes
3. **Queryable**: Can filter by provider OR mode type
4. **Clear semantics**: `(GEMINI, JSON)` vs `GEMINI_JSON`
5. **Easier to extend**: Add provider = register existing mode types
6. **Provider-agnostic**: Can write code using `ModeType.TOOLS` across providers

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
class LegacyMode(Enum):
    """Deprecated flat modes for backward compatibility."""
    GEMINI_JSON = (Provider.GEMINI, ModeType.JSON)
    GEMINI_TOOLS = (Provider.GEMINI, ModeType.TOOLS)
    # ... map all 42 old modes

# Registry supports both
mode_registry.get_handler(LegacyMode.GEMINI_JSON)  # Unwraps to tuple
mode_registry.get_handler(Provider.GEMINI, ModeType.JSON)  # New way
```

---

## Phase 4 Impact

**Phase 4 (Hierarchical Modes)** now has a different role:

**Before**: Introduce hierarchical structure (Provider + ModeType)
**After**: Add rich metadata to already-hierarchical modes

Phase 4 now adds:
- ModeCapability enum
- ModeDescriptor dataclass with metadata
- Capability queries (supports_vision, supports_streaming, etc.)
- Not the hierarchical structure itself (that's now in Phase 1)

---

## Next Steps

1. ✅ Phase 1 document updated with hierarchical design concept
2. ⏳ Update Phase 4 to reflect new role (metadata only)
3. ⏳ During Phase 1 implementation, use hierarchical design throughout
4. ⏳ Implement backward compatibility layer for flat Mode enum

---

**Status**: Architectural decision made, documentation updated
**Implementation**: Use hierarchical design when starting Phase 1
