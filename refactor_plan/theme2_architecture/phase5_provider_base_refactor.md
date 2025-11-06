# Phase 5: Provider Base Class Migration

**Status**: Not Started
**Priority**: P1 (Consolidation)
**Est. Duration**: 4-6 weeks
**Est. Effort**: 20-25 days
**Dependencies**: Theme 1 Phase 2 (Base Classes), Phases 1-2 (Registries)

---

## Overview

Migrate all providers to use base classes from Theme 1 Phase 2, eliminating the remaining 42% code duplication across providers.

### Goal

From **MEASUREMENTS.md**:
- Current: 4,931 lines with 42% duplication (2,085 duplicated lines)
- Target: <10% duplication
- Reduction: 1,600-1,900 lines (32-39%)

---

## Relationship with Theme 1 Phase 2

**This phase IMPLEMENTS the base classes designed in Theme 1 Phase 2.**

**From Theme 1 Phase 2**:
- `BaseProviderHandler` abstract class
- Common interface for all providers
- Shared utility functions

**This phase**:
- Migrates all 19 providers to use base classes
- Eliminates duplication patterns identified in MEASUREMENTS.md
- Consolidates provider-specific code

---

## Implementation Strategy

### Provider Migration Order

**Tier 1** (Highest duplication, migrate first):
1. Gemini (1,060 lines, 70% duplication) → **Save ~740 lines**
2. OpenAI (531 lines, 55% duplication) → **Save ~290 lines**
3. Bedrock (490 lines, 60% duplication) → **Save ~295 lines**
4. Anthropic (462 lines, 50% duplication) → **Save ~230 lines**
5. xAI (462 lines, 60% duplication) → **Save ~280 lines**

**Tier 2** (Medium priority):
6. Cohere (241 lines, 45% duplication) → **Save ~110 lines**

**Tier 3** (Standard providers):
7-13. Mistral, Fireworks, Writer, Cerebras, Genai, Perplexity, Groq (all 40% dup)

**Total potential savings**: 1,600-1,900 lines

---

## Five Major Duplication Patterns to Eliminate

From **MEASUREMENTS.md**:

### 1. Factory Function Structure (650 lines, 13.2%)

**Before** (repeated 13 times):
```python
@overload
def from_anthropic(client: Anthropic, **kwargs) -> Instructor: ...

@overload
def from_anthropic(client: AsyncAnthropic, **kwargs) -> AsyncInstructor: ...

def from_anthropic(client: Anthropic | AsyncAnthropic, **kwargs):
    # Validate mode
    valid_modes = {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_JSON}
    if mode not in valid_modes:
        raise ModeError(...)

    # Dispatch
    ...
```

**After** (single base class method):
```python
class AnthropicProvider(BaseProvider):
    valid_modes = {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_JSON}
    # Overloads and validation handled by base class
```

### 2. Reask Boilerplate (725 lines, 14.7%)

**Before** (repeated 11 times):
```python
def reask_anthropic_tools(kwargs, response, exception):
    kwargs_copy = kwargs.copy()
    messages = kwargs_copy.get("messages", [])
    messages.append({
        "role": "user",
        "content": f"Error: {exception}"
    })
    kwargs_copy["messages"] = messages
    return kwargs_copy
```

**After** (inherit from base):
```python
class AnthropicToolsHandler(BaseModeHandler):
    # Default reask implementation from base
    # Override only if different
```

### 3-5. Message Transformation, Tool/Schema Generation, Handler Registries

All consolidated into base classes.

---

## Detailed Implementation Guide

**This will be fleshed out after Theme 1 Phase 2 is complete.**

**Key tasks**:
1. Ensure Theme 1 Phase 2 base classes are complete
2. Start with Tier 1 providers (highest duplication)
3. Migrate one provider at a time
4. Run full test suite after each migration
5. Document migration patterns for community contributors

---

## Success Criteria

- ☐ Base classes from Theme 1 Phase 2 implemented
- ☐ All 19 providers migrated
- ☐ Duplication reduced from 42% to <10%
- ☐ 1,600-1,900 lines eliminated
- ☐ All tests passing
- ☐ Performance not regressed

---

## Next Steps

1. **Complete Theme 1 Phase 2** first
2. Create detailed migration guide for each provider pattern
3. Migrate Tier 1 providers
4. Document lessons learned
5. Enable community contributions for Tier 2-3

---

**Status**: Ready after Theme 1 Phase 2 + Theme 2 Phases 1-2
**Detailed plan**: Create after base classes are implemented
