# Phase 7: Configuration System (v2.0)

**Status**: Not Started
**Priority**: P2 (v2.0 feature)
**Est. Duration**: 4-6 weeks
**Est. Effort**: 18-22 days
**Dependencies**: Phases 1-6 complete
**Timeline**: v2.0 release only

---

## Overview

Centralized configuration system instead of scattered kwargs. Provides type-safe, discoverable, and well-documented configuration options.

### Current Problem

```python
# Configuration scattered across function parameters
client = instructor.from_openai(
    openai.OpenAI(),
    max_retries=3,  # Where does this come from?
    strict_validation=True,  # What does this do?
    cache_schemas=True,  # Is this even a real parameter?
    some_random_kwarg=True,  # No validation!
)

# No discoverability, no validation, no documentation
```

### Goal

```python
# Type-safe configuration
config = instructor.InstructorConfig(
    max_retries=3,
    strict_validation=True,
    cache_schemas=True,
    debug_mode=False,
)

# Global config
instructor.configure(config)

# Per-client config
client = instructor.from_openai(
    openai.OpenAI(),
    config=config
)

# Environment variable support
config = instructor.InstructorConfig.from_env()
```

---

## Design (from V2_PLAN.md)

```python
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
            debug_mode=os.getenv("INSTRUCTOR_DEBUG", "false").lower() == "true",
            # ... etc
        )
```

---

## Implementation Plan

**Detailed implementation guide will be created closer to v2.0 release.**

**Key tasks**:
1. Define InstructorConfig dataclass
2. Add environment variable loading
3. Add global config management
4. Update all client creation to accept config
5. Deprecate scattered kwargs in favor of config
6. Migration guide

---

## Breaking Changes

- Some kwargs may be removed in favor of config object
- Environment variable naming standardized (INSTRUCTOR_* prefix)

---

## Success Criteria

- ☐ InstructorConfig implemented
- ☐ Environment variable support
- ☐ All clients accept config parameter
- ☐ Documentation complete
- ☐ Migration guide published

---

**Status**: Defer until v2.0 timeline
**Detailed plan**: Create before v2.0 development begins
