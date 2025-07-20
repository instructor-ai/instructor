# Plan to Remove Backward Compatibility Files

## Overview

This document outlines the plan to remove backward compatibility files that were created during the codebase reorganization. These files provide import redirects from old locations to new locations.

## Backward Compatibility Files to Remove

1. **`/instructor/exceptions.py`**
   - Re-exports from: `instructor.core.exceptions`
   - Used by: 8 test files, 4 documentation files

2. **`/instructor/function_calls.py`**
   - Re-exports from: `instructor.processing.function_calls`
   - Used by: 1 test file, 1 example file

3. **`/instructor/multimodal.py`**
   - Re-exports from: `instructor.processing.multimodal`
   - Used by: 3 test files, 15 documentation files, 1 example file
   - Note: Has TODO comment "fix this in v2"

4. **`/instructor/process_response.py`**
   - Re-exports from: `instructor.processing.response`
   - Used by: 2 test files

5. **`/instructor/retry.py`**
   - Re-exports from: `instructor.core.retry`
   - Used by: 2 test files

6. **`/instructor/schema_utils.py`**
   - Re-exports from: `instructor.processing.schema`
   - Used by: 1 test file

7. **`/instructor/validators.py`**
   - Re-exports from: `instructor.validation.async_validators`
   - Used by: 0 files (not found in search)

8. **`/instructor/dsl/validators.py`**
   - Re-exports from: `instructor.validation`
   - Used by: 1 test file

## Migration Mapping

### Old Import → New Import

```python
# Exceptions
from instructor.exceptions import * → from instructor.core.exceptions import *

# Function calls
from instructor.function_calls import * → from instructor.processing.function_calls import *

# Multimodal
from instructor.multimodal import * → from instructor.processing.multimodal import *

# Process response
from instructor.process_response import * → from instructor.processing.response import *

# Retry
from instructor.retry import * → from instructor.core.retry import *

# Schema utils
from instructor.schema_utils import * → from instructor.processing.schema import *

# Validators
from instructor.validators import * → from instructor.validation.async_validators import *

# DSL validators
from instructor.dsl.validators import * → from instructor.validation import *
```

## Files Requiring Updates

### Test Files (18 files)

#### Exceptions imports (8 files):
- `tests/test_function_calls.py`
- `tests/test_exceptions.py` (multiple imports)
- `tests/test_auto_client.py` (3 occurrences)
- `tests/llm/test_vertexai/test_deprecated_async.py`
- `tests/llm/test_cohere/test_retries.py`
- `tests/llm/test_openai/test_retries.py`

#### Multimodal imports (3 files):
- `tests/test_multimodal.py`
- `tests/llm/test_anthropic/test_multimodal.py`
- `tests/llm/test_openai/test_multimodal.py`

#### Process response imports (2 files):
- `tests/test_process_response.py`
- `tests/test_response_model_conversion.py`

#### Retry imports (2 files):
- `tests/test_dict_operations.py`
- `tests/test_dict_operations_validation.py`

#### Schema utils imports (1 file):
- `tests/test_schema_utils.py`

#### Function calls imports (1 file):
- `tests/test_schema_utils.py`

#### DSL validators imports (1 file):
- `tests/llm/test_openai/test_validators.py`

### Documentation Files (19 files)

#### Multimodal imports (15 files):
- `docs/concepts/multimodal.md` (8 occurrences)
- `docs/integrations/genai.md` (4 occurrences)
- `docs/integrations/anthropic.md` (3 occurrences)
- `docs/integrations/openai.md` (3 occurrences)
- `docs/blog/posts/openai-multimodal.md`
- `docs/integrations/mistral.md`
- `docs/examples/audio_extraction.md`

#### Exceptions imports (4 files):
- `docs/concepts/error_handling.md` (7 occurrences)
- `docs/concepts/hooks.md` (2 occurrences)
- `docs/concepts/usage.md`

### Example Files (2 files)

- `examples/openai-audio/run.py` - multimodal import
- `examples/extract-table/run_vision_langsmith.py` - function_calls usage

## Implementation Steps

### Phase 1: Update Test Files
1. Create a new branch: `remove-backward-compat`
2. Update all test file imports to use new paths
3. Run tests to ensure they pass
4. Commit with message: "test: update imports to use new module paths"

### Phase 2: Update Documentation
1. Update all documentation imports to use new paths
2. Build documentation locally to verify
3. Commit with message: "docs: update imports to use new module paths"

### Phase 3: Update Examples
1. Update example file imports
2. Test examples manually
3. Commit with message: "examples: update imports to use new module paths"

### Phase 4: Remove Backward Compatibility Files
1. Delete all 8 backward compatibility files
2. Update `__init__.py` files to remove backward compatibility references
3. Run full test suite
4. Commit with message: "refactor: remove backward compatibility modules"

### Phase 5: Final Verification
1. Run all tests: `pytest tests/`
2. Build documentation: `mkdocs build`
3. Run linters: `ruff check instructor`
4. Create PR with all changes

## Rollback Plan

If issues arise:
1. Keep the branch with updates but don't merge
2. Create a deprecation plan instead:
   - Add deprecation warnings to backward compatibility files
   - Set a timeline for removal (e.g., v2.0)
   - Document the migration in release notes

## Notes

- The `multimodal` module has the most usage (19 files total)
- The `validators.py` backward compatibility file appears unused
- Some files have multiple imports that need updating
- The `__init__.py` files also contain backward compatibility logic that needs cleaning

## Detailed File Checklist

### Test Files to Update

#### Exceptions Module Updates
- [ ] `tests/test_function_calls.py`
  - Change: `from instructor.exceptions import IncompleteOutputException`
  - To: `from instructor.core.exceptions import IncompleteOutputException`

- [ ] `tests/test_exceptions.py`
  - Change: `from instructor.exceptions import (InstructorError, IncompleteOutputException, ...)`
  - To: `from instructor.core.exceptions import (InstructorError, IncompleteOutputException, ...)`
  - Note: Multiple imports on lines 4-13

- [ ] `tests/test_auto_client.py`
  - Line 91: `from instructor.exceptions import ConfigurationError` → `from instructor.core.exceptions import ConfigurationError`
  - Line 100: `from instructor.exceptions import ConfigurationError` → `from instructor.core.exceptions import ConfigurationError`
  - Line 110: `from instructor.exceptions import InstructorRetryException` → `from instructor.core.exceptions import InstructorRetryException`

- [ ] `tests/llm/test_vertexai/test_deprecated_async.py`
  - Change: `from instructor.exceptions import ConfigurationError`
  - To: `from instructor.core.exceptions import ConfigurationError`

- [ ] `tests/llm/test_cohere/test_retries.py`
  - Change: `from instructor.exceptions import InstructorRetryException`
  - To: `from instructor.core.exceptions import InstructorRetryException`

- [ ] `tests/llm/test_openai/test_retries.py`
  - Change: `from instructor.exceptions import InstructorRetryException`
  - To: `from instructor.core.exceptions import InstructorRetryException`

#### Multimodal Module Updates
- [ ] `tests/test_multimodal.py`
  - Change: `from instructor.multimodal import Image, convert_contents, convert_messages`
  - To: `from instructor.processing.multimodal import Image, convert_contents, convert_messages`

- [ ] `tests/llm/test_anthropic/test_multimodal.py`
  - Change: `from instructor.multimodal import Image, PDF, PDFWithCacheControl`
  - To: `from instructor.processing.multimodal import Image, PDF, PDFWithCacheControl`

- [ ] `tests/llm/test_openai/test_multimodal.py`
  - Change: `from instructor.multimodal import Image, Audio`
  - To: `from instructor.processing.multimodal import Image, Audio`

#### Process Response Module Updates
- [ ] `tests/test_process_response.py`
  - Change: `from instructor.process_response import handle_response_model`
  - To: `from instructor.processing.response import handle_response_model`

- [ ] `tests/test_response_model_conversion.py`
  - Change: `from instructor.process_response import handle_response_model`
  - To: `from instructor.processing.response import handle_response_model`

#### Retry Module Updates
- [ ] `tests/test_dict_operations.py`
  - Change: `from instructor.retry import extract_messages`
  - To: `from instructor.core.retry import extract_messages`

- [ ] `tests/test_dict_operations_validation.py`
  - Change: `from instructor.retry import extract_messages`
  - To: `from instructor.core.retry import extract_messages`

#### Schema Utils Module Updates
- [ ] `tests/test_schema_utils.py`
  - Lines 7-11: `from instructor.schema_utils import ...` → `from instructor.processing.schema import ...`
  - Line 12: `from instructor.function_calls import OpenAISchema` → `from instructor.processing.function_calls import OpenAISchema`

#### DSL Validators Module Updates
- [ ] `tests/llm/test_openai/test_validators.py`
  - Change: `from instructor.dsl.validators import llm_validator`
  - To: `from instructor.validation import llm_validator`

### Documentation Files to Update

#### Multimodal Documentation Updates
- [ ] `docs/concepts/multimodal.md`
  - 8 occurrences to update (lines 42, 80, 114, 179, 238, 277, 322, 369)
  - Change all: `from instructor.multimodal import` → `from instructor.processing.multimodal import`

- [ ] `docs/integrations/genai.md`
  - 4 occurrences (lines 287, 339, 385, 438)
  - Change all: `from instructor.multimodal import` → `from instructor.processing.multimodal import`

- [ ] `docs/integrations/anthropic.md`
  - 3 occurrences (lines 176, 231, 273)
  - Change all: `from instructor.multimodal import` → `from instructor.processing.multimodal import`

- [ ] `docs/integrations/openai.md`
  - 3 occurrences (lines 172, 225, 270)
  - Change all: `from instructor.multimodal import` → `from instructor.processing.multimodal import`

- [ ] `docs/blog/posts/openai-multimodal.md`
  - Line 42: `from instructor.multimodal import` → `from instructor.processing.multimodal import`

- [ ] `docs/integrations/mistral.md`
  - Line 314: `from instructor.multimodal import` → `from instructor.processing.multimodal import`

- [ ] `docs/examples/audio_extraction.md`
  - Line 17: `from instructor.multimodal import` → `from instructor.processing.multimodal import`

#### Exceptions Documentation Updates
- [ ] `docs/concepts/error_handling.md`
  - 7 occurrences (lines 15, 127, 152, 165, 186, 211, 239)
  - Change all: `from instructor.exceptions import` → `from instructor.core.exceptions import`

- [ ] `docs/concepts/hooks.md`
  - 2 occurrences (lines 293, 430)
  - Change all: `from instructor.exceptions import` → `from instructor.core.exceptions import`

- [ ] `docs/concepts/usage.md`
  - Line 45: `from instructor.exceptions import` → `from instructor.core.exceptions import`

### Example Files to Update
- [ ] `examples/openai-audio/run.py`
  - Line 4: `from instructor.multimodal import Audio` → `from instructor.processing.multimodal import Audio`

- [ ] `examples/extract-table/run_vision_langsmith.py`
  - Line 18: `instructor.function_calls.Mode.MD_JSON` → `instructor.processing.function_calls.Mode.MD_JSON`

### Backward Compatibility Files to Delete
- [ ] Delete `/instructor/exceptions.py`
- [ ] Delete `/instructor/function_calls.py`
- [ ] Delete `/instructor/multimodal.py`
- [ ] Delete `/instructor/process_response.py`
- [ ] Delete `/instructor/retry.py`
- [ ] Delete `/instructor/schema_utils.py`
- [ ] Delete `/instructor/validators.py`
- [ ] Delete `/instructor/dsl/validators.py`

### Additional Clean-up
- [ ] Update `/instructor/__init__.py` to remove backward compatibility references
- [ ] Check and update any other `__init__.py` files with backward compatibility logic

## Success Criteria

- [ ] All tests pass after import updates
- [ ] Documentation builds without errors
- [ ] No import errors in examples
- [ ] All backward compatibility files removed
- [ ] CI/CD pipeline passes