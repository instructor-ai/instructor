# Instructor Codebase Study Notes

**Author**: Anderson Henrique da Silva
**Location**: Minas Gerais, Brazil
**Date**: 2025-12-11
**Purpose**: Deep dive analysis for open source contribution

---

## 1. Project Overview

**Instructor** is a Python library for extracting structured, validated data from LLMs using Pydantic models. It provides a unified API across 16+ providers with automatic retry logic and validation.

- **Version**: 1.13.0
- **Maintainers**: Jason Liu, Ivan Leo
- **License**: MIT
- **Python**: 3.9 - 4.0

---

## 2. Architecture Summary

```
instructor/
├── core/                 # Core client logic
│   ├── client.py        # Instructor/AsyncInstructor classes
│   ├── hooks.py         # Event system (5 hook types)
│   ├── retry.py         # Tenacity-based retry logic
│   ├── patch.py         # Monkey-patching for providers
│   └── exceptions.py    # Custom exceptions
├── providers/           # 16 LLM provider integrations
│   ├── openai/         # Reference implementation
│   ├── anthropic/      # Claude support
│   ├── google/         # Gemini (legacy)
│   ├── genai/          # Google GenAI (current)
│   └── [13 more...]    # Mistral, Cohere, Groq, etc.
├── processing/          # Response handling
│   ├── response.py     # Central dispatcher (40+ modes)
│   └── schema.py       # JSON schema generation
├── dsl/                 # Domain-specific features
│   ├── partial.py      # Streaming partial objects
│   ├── iterable.py     # Streaming lists
│   └── parallel.py     # Parallel tool calls
├── cache/               # Caching layer
│   └── __init__.py     # BaseCache, AutoCache, DiskCache
├── validation/          # Validators
│   └── llm_validators.py
└── auto_client.py       # Universal `from_provider()` factory
```

---

## 3. Core Components Analysis

### 3.1 Hook System (`core/hooks.py`)

The hook system enables event-driven observability:

```python
class HookName(Enum):
    COMPLETION_KWARGS = "completion:kwargs"      # Before API call
    COMPLETION_RESPONSE = "completion:response"  # After successful response
    COMPLETION_ERROR = "completion:error"        # On API/network errors
    COMPLETION_LAST_ATTEMPT = "completion:last_attempt"  # Final retry failed
    PARSE_ERROR = "parse:error"                  # Validation/JSON parse errors
```

**Key Methods**:
- `hooks.on(hook_name, handler)` - Register handler
- `hooks.emit(hook_name, *args, **kwargs)` - Emit event
- `hooks.off(hook_name, handler)` - Remove handler
- `Hooks.combine(*hooks)` - Merge multiple hook instances

**Usage Pattern**:
```python
client = instructor.from_provider("openai/gpt-4")
client.on("completion:response", lambda r: print(f"Tokens: {r.usage}"))
```

### 3.2 Cache System (`cache/__init__.py`)

Current implementation provides transparent response caching:

```python
class BaseCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...

class AutoCache(BaseCache):  # In-memory LRU
class DiskCache(BaseCache):  # diskcache wrapper
```

**Cache Key Generation** (`make_cache_key`):
- Combines: model, messages, mode, response_model schema
- Uses SHA-256 hash for fixed-length keys

**Current Logging** (lines 237-277):
```python
logger.debug("cache hit: %s", key)   # In load_cached_response()
logger.debug("cache store: %s", key)  # In store_cached_response()
```

**Gap Identified**: No hook emissions for cache events - only debug logging.

### 3.3 Retry Logic (`core/retry.py`)

Uses Tenacity for configurable retry:

```python
def retry_sync(func, response_model, args, kwargs, ...):
    for attempt in max_retries:
        try:
            hooks.emit_completion_arguments(*args, **kwargs)
            response = func(*args, **kwargs)
            hooks.emit_completion_response(response)
            return process_response(response, ...)
        except (ValidationError, JSONDecodeError) as e:
            hooks.emit_parse_error(e)
            kwargs = handle_reask_kwargs(...)  # Reask pattern
            raise e
```

### 3.4 Mode System (`mode.py`)

40+ modes for different provider/format combinations:

| Category | Modes |
|----------|-------|
| OpenAI | TOOLS, PARALLEL_TOOLS, JSON, JSON_SCHEMA, TOOLS_STRICT |
| Anthropic | ANTHROPIC_TOOLS, ANTHROPIC_JSON, ANTHROPIC_REASONING_TOOLS |
| Google | GEMINI_JSON, GEMINI_TOOLS, GENAI_TOOLS, GENAI_STRUCTURED_OUTPUTS |
| Others | MISTRAL_TOOLS, COHERE_TOOLS, BEDROCK_TOOLS, etc. |

---

## 4. Issue #1882 Analysis: Cache Observability Hooks

### 4.1 Current State

Cache operations only log via `logging.debug()`:
- `instructor/cache/__init__.py:237` - cache hit
- `instructor/cache/__init__.py:277` - cache store

No integration with the hook system despite hooks being the standard observability pattern.

### 4.2 Proposed Solution

Add new hook types for cache events:

```python
# In core/hooks.py
class HookName(Enum):
    # ... existing hooks ...
    CACHE_HIT = "cache:hit"
    CACHE_MISS = "cache:miss"
```

Emit hooks in cache operations:

```python
# In cache/__init__.py or core/patch.py
def load_cached_response(cache, key, response_model, hooks=None):
    cached = cache.get(key)
    if cached is None:
        if hooks:
            hooks.emit(HookName.CACHE_MISS, key=key)
        return None
    # ... existing logic ...
    if hooks:
        hooks.emit(HookName.CACHE_HIT, key=key, cached=obj)
    return obj
```

### 4.3 Implementation Considerations

1. **Backward Compatibility**: Hooks parameter should be optional
2. **Performance**: Minimal overhead when no handlers registered
3. **Consistency**: Follow existing hook patterns (emit with kwargs)
4. **Testing**: Add unit tests for new hooks
5. **Documentation**: Update hook documentation

### 4.4 Files to Modify

1. `instructor/core/hooks.py` - Add CACHE_HIT, CACHE_MISS to HookName enum
2. `instructor/cache/__init__.py` - Emit hooks in load/store functions
3. `instructor/core/patch.py` - Pass hooks to cache functions
4. `tests/test_cache_hooks.py` - New test file
5. `docs/concepts/hooks.md` - Document new hooks

---

## 5. Codebase Patterns

### 5.1 Provider Implementation Pattern

Each provider follows this structure:

```
providers/provider_name/
├── __init__.py           # Minimal exports
├── client.py            # from_<provider>() factory
└── utils.py             # handle_<mode>(), reask_<mode>()
```

### 5.2 Response Processing Flow

```
1. User calls client.create(response_model=Model)
2. core/patch.py wraps the call
3. hooks.emit_completion_arguments()
4. Provider API called
5. hooks.emit_completion_response()
6. processing/response.py dispatches by Mode
7. Provider-specific handler extracts data
8. Pydantic validation
9. Return validated model or retry with reask
```

### 5.3 Testing Patterns

```python
# Unit tests - no API calls
@pytest.mark.unit
def test_cache_key_generation():
    ...

# Integration tests - may need API keys
@pytest.mark.integration
def test_openai_extraction():
    ...

# LLM tests - actual API calls
@pytest.mark.llm
def test_real_extraction():
    ...
```

---

## 6. Dependencies

### Core
- `openai>=2.0.0` - Reference client
- `pydantic>=2.8.0` - Data validation
- `tenacity>=8.2.3` - Retry logic
- `jiter>=0.6.1` - Fast JSON parsing
- `diskcache>=5.6.3` - Disk caching

### Optional Providers
- `anthropic==0.71.0` - Claude
- `google-genai>=1.5.0` - Gemini
- `mistralai>=1.5.1` - Mistral
- `boto3>=1.34.0` - AWS Bedrock
- `groq>=0.4.2` - Groq

---

## 7. Contribution Checklist

- [ ] Fork and clone repository
- [ ] Set up development environment with `uv pip install -e ".[dev]"`
- [ ] Run `pre-commit install`
- [ ] Create feature branch from main
- [ ] Implement changes following existing patterns
- [ ] Write tests (unit + integration)
- [ ] Run `pytest tests/ -k 'not llm'`
- [ ] Update documentation if needed
- [ ] Create PR with conventional commit message

---

## 8. Next Steps

1. **Day 1**: Fork + clone + study codebase (DONE)
2. **Day 2-3**: Comment on issue #1882 to claim it
3. **Day 3-4**: Implement cache hooks following existing patterns
4. **Day 4-5**: Write tests + Submit PR

---

## 9. Issue Comment Draft (Post on Day 2-3)

**Command to post:**
```bash
cd /home/anderson-henrique/Documentos/opensource-contributions/instructor
gh issue comment 1882 --body "$(cat <<'EOF'
Hi! I'd like to work on this issue.

I've been studying the codebase and the hook system architecture. My proposed approach:

- Add `CACHE_HIT` and `CACHE_MISS` events to the `HookName` enum in `core/hooks.py`
- Emit these hooks in `load_cached_response()` and `store_cached_response()` functions
- Ensure backward compatibility by making the hooks parameter optional
- Add corresponding emit methods (`emit_cache_hit`, `emit_cache_miss`) following the existing pattern

This will enable users to track cache performance metrics like hit ratio and latency improvements as mentioned in the issue.

Will submit a PR in the next few days.
EOF
)"
```

---

## References

- Repository: https://github.com/jxnl/instructor
- Issue #1882: Cache hit/miss event hooks
- CONTRIBUTING.md: Development guidelines
- NEW_PROVIDER_AGENT_INSTRUCTIONS.md: Provider implementation guide
