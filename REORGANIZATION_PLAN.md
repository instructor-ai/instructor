# Instructor Package Reorganization Plan

## COMPLETION STATUS: ✅ COMPLETED

### Summary of Changes
- ✅ Moved all 13 provider files to organized subdirectories under `providers/`
- ✅ Created modular structure: `core/`, `providers/`, `processing/`, `validation/`
- ✅ Fixed circular import dependencies
- ✅ Maintained backward compatibility with compatibility modules
- ✅ All tests passing

### Additional Work Done
- Created `processing/validators.py` to resolve circular imports
- Moved `AsyncValidationError` to `core/exceptions.py`
- Created backward compatibility modules: `multimodal.py`, `dsl/validators.py`
- Updated all imports throughout the codebase

## Overview
This document outlines a detailed plan to reorganize the instructor package from its current flat structure to a more modular, maintainable architecture.

## Current Structure Analysis

### Issues Identified
1. **Provider Sprawl**: 13 provider client files (`client_*.py`) in the root directory
2. **Split Provider Logic**: Client files in root, utilities in `/utils/` subfolder
3. **Mixed Concerns**: Core processing logic intermingled with provider-specific imports
4. **Scattered Schema Utilities**: Schema generation split across multiple files
5. **Fragmented Validation**: Validation code in multiple locations
6. **Monolithic Mode File**: Single file containing 40+ mode definitions

### Files to Migrate

#### Provider Client Files (13 files)
- [x] `client_anthropic.py`
- [x] `client_bedrock.py`
- [x] `client_cerebras.py`
- [x] `client_cohere.py`
- [x] `client_fireworks.py`
- [x] `client_gemini.py`
- [x] `client_genai.py`
- [x] `client_groq.py`
- [x] `client_mistral.py`
- [x] `client_perplexity.py`
- [x] `client_vertexai.py`
- [x] `client_writer.py`
- [x] `client_xai.py`

#### Provider Utility Files (13 files)
- [x] `utils/anthropic.py`
- [x] `utils/bedrock.py`
- [x] `utils/cerebras.py`
- [x] `utils/cohere.py`
- [x] `utils/fireworks.py`
- [x] `utils/google.py`
- [x] `utils/mistral.py`
- [x] `utils/openai.py`
- [x] `utils/perplexity.py`
- [x] `utils/writer.py`
- [x] `utils/xai.py`
- [x] `utils/core.py` (kept in utils)
- [x] `utils/providers.py` (kept in utils)

#### Core Framework Files
- [x] `client.py` → `core/client.py`
- [x] `patch.py` → `core/patch.py`
- [x] `retry.py` → `core/retry.py`
- [x] `hooks.py` → `core/hooks.py`
- [x] `exceptions.py` → `core/exceptions.py`

#### Processing Files
- [x] `process_response.py` → `processing/response.py`
- [x] `function_calls.py` → `processing/function_calls.py`
- [x] `schema_utils.py` → `processing/schema.py`
- [x] `multimodal.py` → `processing/multimodal.py`

#### Validation Files
- [x] `validators.py` → `validation/async_validators.py`
- [x] `dsl/validators.py` → `validation/llm_validators.py`

#### Mode Definitions
- [ ] `mode.py` → `modes/mode.py` (or keep in root initially)

## New Directory Structure

```
instructor/
├── core/                    # Core framework components
│   ├── __init__.py
│   ├── client.py           # Instructor, AsyncInstructor base classes
│   ├── patch.py            # Core patching logic
│   ├── retry.py            # Retry mechanisms
│   ├── hooks.py            # Event hooks
│   └── exceptions.py       # Core exceptions
│
├── providers/              # All provider-specific code
│   ├── __init__.py        # Exports all from_* functions
│   ├── anthropic/
│   │   ├── __init__.py
│   │   ├── client.py      # from_anthropic function
│   │   └── utils.py       # Anthropic-specific utilities
│   ├── openai/
│   │   ├── __init__.py
│   │   ├── client.py      # from_openai function  
│   │   └── utils.py       # OpenAI-specific utilities
│   ├── bedrock/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── utils.py
│   ├── cerebras/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── utils.py
│   ├── cohere/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── utils.py
│   ├── fireworks/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── utils.py
│   ├── gemini/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── utils.py
│   ├── genai/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── utils.py
│   ├── groq/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── utils.py
│   ├── mistral/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── utils.py
│   ├── perplexity/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── utils.py
│   ├── vertexai/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── utils.py
│   ├── writer/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── utils.py
│   └── xai/
│       ├── __init__.py
│       ├── client.py
│       └── utils.py
│
├── processing/             # Request/response processing
│   ├── __init__.py
│   ├── response.py        # Main process_response logic
│   ├── schema.py          # Schema generation utilities
│   ├── function_calls.py  # Function/tool schema generation
│   └── multimodal.py      # Multimodal message handling
│
├── validation/             # All validation logic
│   ├── __init__.py
│   ├── async_validators.py # Async validation framework
│   └── llm_validators.py   # LLM-based validators
│
├── modes/                  # Mode definitions (optional for phase 2)
│   ├── __init__.py
│   └── mode.py            # Mode enum definitions
│
├── dsl/                    # Keep existing DSL components
│   ├── __init__.py
│   ├── citation.py
│   ├── iterable.py
│   ├── maybe.py
│   ├── parallel.py
│   ├── partial.py
│   └── simple_type.py     # Note: validators.py will be moved
│
├── batch/                  # Keep existing batch processing
├── cli/                    # Keep existing CLI tools
├── cache/                  # Keep existing cache
├── _types/                 # Keep existing types
├── utils/                  # General utilities only
│   ├── __init__.py
│   ├── core.py            # Core utilities
│   └── providers.py       # Provider enum
│
├── __init__.py            # Updated imports
├── auto_client.py         # Unified provider access
├── distil.py              # Keep in root
├── templating.py          # Keep in root
├── models.py              # Keep in root
└── py.typed               # Keep in root
```

## Migration Steps

### Phase 1: Create Directory Structure
- [ ] Create `core/` directory with `__init__.py`
- [ ] Create `providers/` directory with `__init__.py`
- [ ] Create provider subdirectories:
  - [ ] `providers/anthropic/` with `__init__.py`
  - [ ] `providers/openai/` with `__init__.py`
  - [ ] `providers/bedrock/` with `__init__.py`
  - [ ] `providers/cerebras/` with `__init__.py`
  - [ ] `providers/cohere/` with `__init__.py`
  - [ ] `providers/fireworks/` with `__init__.py`
  - [ ] `providers/gemini/` with `__init__.py`
  - [ ] `providers/genai/` with `__init__.py`
  - [ ] `providers/groq/` with `__init__.py`
  - [ ] `providers/mistral/` with `__init__.py`
  - [ ] `providers/perplexity/` with `__init__.py`
  - [ ] `providers/vertexai/` with `__init__.py`
  - [ ] `providers/writer/` with `__init__.py`
  - [ ] `providers/xai/` with `__init__.py`
- [ ] Create `processing/` directory with `__init__.py`
- [ ] Create `validation/` directory with `__init__.py`
- [ ] Create `modes/` directory with `__init__.py` (optional)

### Phase 2: Move Core Framework Files
- [ ] Move `client.py` → `core/client.py`
  - [ ] Update imports in moved file
  - [ ] Create compatibility import in root `__init__.py`
- [ ] Move `patch.py` → `core/patch.py`
  - [ ] Update imports in moved file
  - [ ] Create compatibility import in root `__init__.py`
- [ ] Move `retry.py` → `core/retry.py`
  - [ ] Update imports in moved file
  - [ ] Create compatibility import in root `__init__.py`
- [ ] Move `hooks.py` → `core/hooks.py`
  - [ ] Update imports in moved file
  - [ ] Create compatibility import in root `__init__.py`
- [ ] Move `exceptions.py` → `core/exceptions.py`
  - [ ] Update imports in moved file
  - [ ] Create compatibility import in root `__init__.py`
- [ ] Update `core/__init__.py` to export all core components

### Phase 3: Move Provider Files
For each provider (anthropic, openai, bedrock, etc.):

#### Anthropic
- [ ] Move `client_anthropic.py` → `providers/anthropic/client.py`
  - [ ] Update imports in file
- [ ] Move `utils/anthropic.py` → `providers/anthropic/utils.py`
  - [ ] Update imports in file
- [ ] Update `providers/anthropic/__init__.py` to export `from_anthropic`
- [ ] Update all files importing from `client_anthropic` or `utils.anthropic`

#### OpenAI
- [ ] Move `client_openai.py` → `providers/openai/client.py` (Note: doesn't exist, might be in patch.py)
  - [ ] Update imports in file
- [ ] Move `utils/openai.py` → `providers/openai/utils.py`
  - [ ] Update imports in file
- [ ] Update `providers/openai/__init__.py` to export OpenAI functions
- [ ] Update all files importing from OpenAI utilities

#### Bedrock
- [ ] Move `client_bedrock.py` → `providers/bedrock/client.py`
  - [ ] Update imports in file
- [ ] Move `utils/bedrock.py` → `providers/bedrock/utils.py`
  - [ ] Update imports in file
- [ ] Update `providers/bedrock/__init__.py` to export `from_bedrock`
- [ ] Update all files importing from `client_bedrock` or `utils.bedrock`

#### Cerebras
- [ ] Move `client_cerebras.py` → `providers/cerebras/client.py`
  - [ ] Update imports in file
- [ ] Move `utils/cerebras.py` → `providers/cerebras/utils.py`
  - [ ] Update imports in file
- [ ] Update `providers/cerebras/__init__.py` to export `from_cerebras`
- [ ] Update all files importing from `client_cerebras` or `utils.cerebras`

#### Cohere
- [ ] Move `client_cohere.py` → `providers/cohere/client.py`
  - [ ] Update imports in file
- [ ] Move `utils/cohere.py` → `providers/cohere/utils.py`
  - [ ] Update imports in file
- [ ] Update `providers/cohere/__init__.py` to export `from_cohere`
- [ ] Update all files importing from `client_cohere` or `utils.cohere`

#### Fireworks
- [ ] Move `client_fireworks.py` → `providers/fireworks/client.py`
  - [ ] Update imports in file
- [ ] Move `utils/fireworks.py` → `providers/fireworks/utils.py`
  - [ ] Update imports in file
- [ ] Update `providers/fireworks/__init__.py` to export `from_fireworks`
- [ ] Update all files importing from `client_fireworks` or `utils.fireworks`

#### Gemini
- [ ] Move `client_gemini.py` → `providers/gemini/client.py`
  - [ ] Update imports in file
- [ ] Move `utils/google.py` → `providers/gemini/utils.py`
  - [ ] Update imports in file
- [ ] Update `providers/gemini/__init__.py` to export `from_gemini`
- [ ] Update all files importing from `client_gemini` or `utils.google`

#### GenAI
- [ ] Move `client_genai.py` → `providers/genai/client.py`
  - [ ] Update imports in file
- [ ] Create `providers/genai/utils.py` if needed
- [ ] Update `providers/genai/__init__.py` to export `from_genai`
- [ ] Update all files importing from `client_genai`

#### Groq
- [ ] Move `client_groq.py` → `providers/groq/client.py`
  - [ ] Update imports in file
- [ ] Create `providers/groq/utils.py` if needed
- [ ] Update `providers/groq/__init__.py` to export `from_groq`
- [ ] Update all files importing from `client_groq`

#### Mistral
- [ ] Move `client_mistral.py` → `providers/mistral/client.py`
  - [ ] Update imports in file
- [ ] Move `utils/mistral.py` → `providers/mistral/utils.py`
  - [ ] Update imports in file
- [ ] Update `providers/mistral/__init__.py` to export `from_mistral`
- [ ] Update all files importing from `client_mistral` or `utils.mistral`

#### Perplexity
- [ ] Move `client_perplexity.py` → `providers/perplexity/client.py`
  - [ ] Update imports in file
- [ ] Move `utils/perplexity.py` → `providers/perplexity/utils.py`
  - [ ] Update imports in file
- [ ] Update `providers/perplexity/__init__.py` to export `from_perplexity`
- [ ] Update all files importing from `client_perplexity` or `utils.perplexity`

#### VertexAI
- [ ] Move `client_vertexai.py` → `providers/vertexai/client.py`
  - [ ] Update imports in file
- [ ] Create `providers/vertexai/utils.py` if needed
- [ ] Update `providers/vertexai/__init__.py` to export `from_vertexai`
- [ ] Update all files importing from `client_vertexai`

#### Writer
- [ ] Move `client_writer.py` → `providers/writer/client.py`
  - [ ] Update imports in file
- [ ] Move `utils/writer.py` → `providers/writer/utils.py`
  - [ ] Update imports in file
- [ ] Update `providers/writer/__init__.py` to export `from_writer`
- [ ] Update all files importing from `client_writer` or `utils.writer`

#### xAI
- [ ] Move `client_xai.py` → `providers/xai/client.py`
  - [ ] Update imports in file
- [ ] Move `utils/xai.py` → `providers/xai/utils.py`
  - [ ] Update imports in file
- [ ] Update `providers/xai/__init__.py` to export `from_xai`
- [ ] Update all files importing from `client_xai` or `utils.xai`

### Phase 4: Move Processing Files
- [ ] Move `process_response.py` → `processing/response.py`
  - [ ] Update all provider imports to use new paths
  - [ ] Update imports in files that use process_response
- [ ] Move `function_calls.py` → `processing/function_calls.py`
  - [ ] Update imports in moved file
  - [ ] Update imports in files that use function_calls
- [ ] Move `schema_utils.py` → `processing/schema.py`
  - [ ] Update imports in moved file
  - [ ] Update imports in files that use schema_utils
- [ ] Move `multimodal.py` → `processing/multimodal.py`
  - [ ] Update imports in moved file
  - [ ] Update imports in files that use multimodal
- [ ] Update `processing/__init__.py` to export all processing functions

### Phase 5: Move Validation Files
- [ ] Move `validators.py` → `validation/async_validators.py`
  - [ ] Update imports in moved file
  - [ ] Update imports in files that use validators
- [ ] Move `dsl/validators.py` → `validation/llm_validators.py`
  - [ ] Update imports in moved file
  - [ ] Update imports in files that use dsl.validators
- [ ] Update `validation/__init__.py` to export all validators
- [ ] Update `dsl/__init__.py` to remove validators export

### Phase 6: Update Main Package Imports
- [ ] Update root `__init__.py` to maintain backward compatibility:
  ```python
  # Maintain backward compatibility
  from .core.client import Instructor, AsyncInstructor, from_openai
  from .core.patch import apatch, patch
  from .core.retry import reask_messages, retry_async, retry_sync
  from .core.hooks import *
  from .core.exceptions import *
  
  # Provider imports
  from .providers import *  # This will import all from_* functions
  
  # Auto client
  from .auto_client import from_provider
  
  # Keep existing exports
  from .distil import FinetuneFormat, Instructions
  from .dsl import *
  from .models import *
  # etc...
  ```

### Phase 7: Update auto_client.py
- [ ] Update import statements to use new provider paths
- [ ] Update dynamic imports to look in `providers/` subdirectories
- [ ] Test that `from_provider()` still works correctly

### Phase 8: Update Tests
- [ ] Update test imports for core components
- [ ] Update test imports for provider-specific code
- [ ] Update test imports for processing functions
- [ ] Update test imports for validation
- [ ] Run full test suite to ensure nothing is broken

### Phase 9: Update Documentation References
- [ ] Update CLAUDE.md to reflect new structure
- [ ] Update any documentation that references file paths
- [ ] Update contributing guidelines with new structure

### Phase 10: Cleanup and Verification
- [ ] Remove old files that have been moved
- [ ] Verify all imports are working
- [ ] Run pyright/mypy to check types
- [ ] Run ruff to check for import issues
- [ ] Run full test suite again
- [ ] Test example scripts to ensure they still work

## Import Update Checklist

### Files that need import updates:
- [ ] `auto_client.py` - Update all provider imports
- [ ] `batch/processor.py` - Update provider imports
- [ ] `batch/providers/*.py` - Update base imports
- [ ] All test files in `tests/llm/test_*/` - Update imports
- [ ] Example files that import from instructor
- [ ] Any file importing `process_response`
- [ ] Any file importing `function_calls`
- [ ] Any file importing `schema_utils`
- [ ] Any file importing validators

## Backward Compatibility Strategy

To maintain backward compatibility:

1. **Root __init__.py exports**: Keep all current exports but source from new locations
2. **Compatibility imports**: For critical paths, consider adding compatibility imports:
   ```python
   # In root __init__.py
   # Backward compatibility
   from .core.client import Instructor as Instructor
   from .providers.anthropic import from_anthropic as from_anthropic
   # etc...
   ```

3. **Deprecation warnings** (Phase 2): Add deprecation warnings for direct imports from old locations
4. **Grace period**: Maintain compatibility imports for 2-3 releases before removal

## Testing Plan

### Pre-migration Testing
- [ ] Run full test suite and save results
- [ ] Document any existing test failures
- [ ] Run coverage report and save baseline

### During Migration Testing
- [ ] After each phase, run relevant tests
- [ ] Fix any import errors immediately
- [ ] Document any behavior changes

### Post-migration Testing
- [ ] Run full test suite
- [ ] Compare with pre-migration results
- [ ] Run all example scripts
- [ ] Test CLI functionality
- [ ] Test with real API calls for each provider

## Rollback Plan

If issues arise:
1. Git reset to pre-migration commit
2. Restore from backup branch
3. Document issues encountered for next attempt

## Success Criteria

- [ ] All tests pass
- [ ] No changes to public API
- [ ] All examples work
- [ ] Documentation is updated
- [ ] Code is more maintainable and organized
- [ ] New provider additions are simpler

## Notes

- Consider using automated refactoring tools for import updates
- Make incremental commits after each successful phase
- Keep a migration log of issues encountered
- Consider creating a script to automate import updates