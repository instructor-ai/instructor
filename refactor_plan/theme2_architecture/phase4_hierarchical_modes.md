# Phase 4: Mode Metadata System (v2.0)

**Status**: Not Started
**Priority**: P2 (v2.0 feature)
**Est. Duration**: 4-6 weeks
**Est. Effort**: 15-20 days
**Dependencies**: Phases 1-3 complete
**Timeline**: v2.0 release only (breaking change)

**Note**: **Mode registry with (Provider, Mode) tuples is in Phase 1.** This phase adds rich metadata on top of the existing modes.

---

## Overview

Add rich metadata to modes (from Phase 1) including capabilities, feature flags, and queryable attributes. This is a **breaking change** requiring v2.0.

### Current Problem (After Phase 1)

Phase 1 gives us mode registry:
```python
# Phase 1 gives us:
mode = (Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS)

# But no metadata:
# - Does this support streaming?
# - Does this support vision?
# - What's the API field name?
# - What capabilities does it have?
```

### Goal (Phase 4)

Add metadata to mode combinations:

```python
# Get metadata for a mode combination
meta = mode_registry.get_metadata(Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS)

meta.supports_streaming  # True
meta.supports_vision  # True
meta.supports_parallel  # True
meta.api_field_name  # "tools"
meta.capabilities  # {TOOLS, STREAMING, VISION, PARALLEL}

# Query by capability
streaming_combos = mode_registry.query(supports_streaming=True)
# → [(ANTHROPIC, TOOLS), (OPENAI, TOOLS), (GEMINI, JSON), ...]

vision_combos = mode_registry.query(supports_vision=True)
# → [(ANTHROPIC, TOOLS), (OPENAI, TOOLS), (GEMINI, TOOLS)]
```

---

## Key Design

From V2_PLAN.md:

```python
from enum import Enum
from dataclasses import dataclass

class ModeCapability(Enum):
    TOOLS = "tools"
    JSON = "json"
    STREAMING = "streaming"
    PARALLEL = "parallel"
    REASONING = "reasoning"
    VISION = "vision"
    AUDIO = "audio"

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
    """Hierarchical mode system with metadata."""

    OPENAI_TOOLS = ModeDescriptor(
        provider="openai",
        capabilities={ModeCapability.TOOLS, ModeCapability.STREAMING},
        supports_streaming=True,
        supports_vision=True,
        supports_audio=False,
        api_field_name="tools"
    )

    ANTHROPIC_TOOLS = ModeDescriptor(
        provider="anthropic",
        capabilities={ModeCapability.TOOLS, ModeCapability.STREAMING, ModeCapability.VISION},
        supports_streaming=True,
        supports_vision=True,
        supports_audio=False,
        api_field_name="tools"
    )

    # ... all 42 modes with full metadata
```

---

## Implementation Plan

**Detailed implementation guide will be created closer to v2.0 release.**

**Key tasks**:
1. Define ModeCapability enum
2. Create ModeDescriptor dataclass
3. Migrate all 42 modes to new structure
4. Update mode registry to use descriptors
5. Add query methods (`has_capability`, `supports_*`)
6. Update all provider code
7. Migration guide for users

---

## Breaking Changes

- Mode enum values change from strings to ModeDescriptor
- Mode comparison changes (`.value` needed for string comparison)
- Some mode names may be renamed for consistency

---

## Success Criteria

- ☐ All modes have complete metadata
- ☐ Capability queries work
- ☐ Documentation auto-generated from metadata
- ☐ Migration guide complete
- ☐ All tests passing

---

**Status**: Defer until v2.0 timeline
**Detailed plan**: Create before v2.0 development begins
