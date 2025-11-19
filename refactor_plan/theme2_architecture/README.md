# Theme 2: Architecture Modernization

**Status**: Not Started
**Estimated Duration**: 6-12 months
**Dependencies**: Theme 1, Phase 2 (Base Classes) recommended
**Priority**: HIGH (enables v2.0)

---

## Overview

Introduce registry-based architecture for modes and providers, eliminate hardcoded dependencies, enable plugin system.

## Phases

### Phase 1: Mode Handler Registry (4-6 weeks)
**Document**: `phase1_mode_registry.md` (TODO: Create)

Decouple mode handling with dynamic registration system.

**Current**: 3 hardcoded dispatch dictionaries (37 modes)
**Target**: Single registry with lazy loading

---

### Phase 2: Provider Registry (4-6 weeks)
**Document**: `phase2_provider_registry.md` (TODO: Create)

Eliminate 924-line if/elif chain in auto_client.py.

**Current**: 19 providers hardcoded with 85% duplication
**Target**: Registry-based with ~100 lines

---

### Phase 3: Lazy Provider Loading (2-3 weeks)
**Document**: `phase3_lazy_loading.md` (TODO: Create)

Only load providers when used.

**Current**: All 11 provider utils loaded on import (~500ms)
**Target**: Lazy loading (~50ms import time)

---

### Phase 4: Hierarchical Mode System (6-8 weeks, v2.0)
**Document**: `phase4_hierarchical_modes.md` (TODO: Create)

Add rich metadata to modes (capabilities, provider mapping, etc).

**Current**: Flat enum with no metadata
**Target**: Modes with queryable capabilities

---

### Phase 5: Provider Base Class Migration (4-6 weeks)
**Document**: `phase5_provider_base_refactor.md` (TODO: Create)

Migrate all providers to use base classes from Theme 1.

**Current**: 42% duplication across providers
**Target**: <10% duplication

---

### Phase 6: Auto Client Refactor (2-3 weeks)
**Document**: `phase6_auto_client_refactor.md` (TODO: Create)

Use registries to simplify auto_client.py.

**Current**: 924 lines
**Target**: <200 lines

---

### Phase 7: Configuration System (4-6 weeks, v2.0)
**Document**: `phase7_configuration_system.md` (TODO: Create)

Centralized configuration instead of scattered kwargs.

**Current**: Configuration spread across function parameters
**Target**: InstructorConfig class with env var support

---

## Success Criteria

- Mode registry implemented and all providers migrated
- Provider registry implemented
- Lazy loading reduces import time by 90%
- auto_client.py reduced from 924 to <200 lines
- All providers using base classes
- Configuration system in place (v2.0)

---

## Dependencies

**Requires**: Theme 1, Phase 2 (Base Classes)
**Enables**: Theme 5 (Plugin Ecosystem)

---

**Detailed phase documents coming soon. Start with Theme 1 first.**
