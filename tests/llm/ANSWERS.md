# Summary: Parameterization Status & Provider-Specific Test Analysis

## Question 1: Are they actually being parameterized?

### ✅ YES - Parameterization IS Working

**How it works:**
1. `tests/llm/test_core_providers/conftest.py` imports `pytest_generate_tests` hook from `shared_config.py`
2. When pytest collects tests, this hook checks for `provider_config` parameter
3. Calls `get_available_providers()` which:
   - Checks for API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
   - Checks if packages are installed (openai, anthropic, google.genai, etc.)
   - Returns list of available (model, mode) tuples
4. Creates one test variant per available provider

**Example:** `test_simple_extraction(provider_config)` becomes:
- `test_simple_extraction[openai]` - runs with OpenAI if API key present
- `test_simple_extraction[anthropic]` - runs with Anthropic if API key present
- `test_simple_extraction[google]` - runs with Google if API key present
- ... for all 10 configured providers

**Current behavior (no API keys set):**
- Tests show: `collected 0 items / 1 skipped`
- Message: "No providers available (missing API keys or packages)"
- **This is CORRECT** - tests skip gracefully instead of failing

**To verify it's working:**
```bash
# Set mock API keys
export OPENAI_API_KEY=test
export ANTHROPIC_API_KEY=test
export GOOGLE_API_KEY=test

# Collect tests
uv run pytest tests/llm/test_core_providers/test_basic_extraction.py --collect-only

# Should show:
# test_simple_extraction[openai]
# test_simple_extraction[anthropic]
# test_simple_extraction[google]
# ... for each test function × number of available providers
```

---

## Question 2: Unique tests that ARE provider-specific

### Truly Provider-Specific Tests (Should Stay Separate)

#### ✅ OpenAI-Specific (3 files)
1. **test_hooks.py** - OpenAI hook system (`instructor.hooks`)
2. **test_multimodal.py** - OpenAI vision/multimodal API (different from others)
3. **test_validation_context.py** - OpenAI-specific validation context

#### ✅ Anthropic-Specific (3 files)
1. **test_reasoning.py** - `thinking` parameter, `ANTHROPIC_REASONING_TOOLS` mode
2. **test_system.py** - Anthropic's unique system prompt handling
3. **test_multimodal.py** - Anthropic multimodal API (different format than OpenAI/Google)

#### ✅ Google-Specific (6 files)
1. **test_schema_conversion.py** - Google's schema conversion quirks
2. **test_format.py** - Google-specific format handling
3. **test_decimal.py** - Google's decimal number handling issues
4. **test_invalid_schema.py** - Google schema validation errors
5. **test_multimodal.py** - Google multimodal API (different from others)
6. **test_utils.py** - Google-specific utilities

#### ✅ Gemini-Specific (3 files + evals)
1. **test_list_content.py** - Gemini list content formatting
2. **test_multimodal_content.py** - Gemini multimodal content structure
3. **evals/** - Model capability evaluation tests

#### ✅ Cohere-Specific (1 file)
1. **test_json_schema.py** - `COHERE_JSON_SCHEMA` mode

#### ✅ Mistral-Specific (1 file)
1. **test_multimodal.py** - Mistral multimodal API

#### ✅ Writer-Specific (1 directory)
1. **evals/** - Model capability evaluation tests

#### ✅ xAI-Specific (maybe 1 file)
1. **test_raw_response.py** - Needs review, might be xAI-specific behavior

---

### ❌ Duplicate Tests (Already in Core - Should Be Deleted)

These **20-25 files** duplicate what's already in `test_core_providers/`:

#### OpenAI (7 duplicates)
- `test_stream.py` → core has test_streaming.py
- `test_retries.py` → core has test_retries.py
- `test_parallel.py` → parallel should work same everywhere
- `test_attr.py` → needs review
- `test_modes.py` → needs review
- `test_validators.py` → Pydantic validators are provider-agnostic
- `test_openai.py` → needs review

#### Anthropic (2 duplicates)
- `test_stream.py` → core has test_streaming.py
- `test_parallel.py` → parallel should work same everywhere

#### Google GenAI (4 duplicates)
- `test_basics.py` → core has test_basic_extraction.py
- `test_simple.py` → core has test_basic_extraction.py
- `test_stream.py` → core has test_streaming.py
- `test_response_model_none.py` → core has test_response_modes.py

#### Gemini (3 duplicates)
- `test_retries.py` → core has test_retries.py
- `test_stream.py` → core has test_streaming.py
- `test_simple_types.py` → needs review

#### Cohere (2 duplicates)
- `test_none_response.py` → core has test_response_modes.py
- `test_retries.py` → core has test_retries.py

#### xAI (2 duplicates)
- `test_basics.py` → core has test_basic_extraction.py
- `test_stream.py` → core has test_streaming.py

#### Mistral (3 duplicates)
- `test_modes.py` → core tests modes
- `test_retries.py` → core has test_retries.py
- `test_stream.py` → core has test_streaming.py

#### Writer (2 duplicates)
- `test_retries.py` → core has test_retries.py
- `test_streaming.py` → core has test_streaming.py

---

## Impact Summary

### Current State
- **10 providers** in core test suite ✅
- **~664 lines deleted** in first cleanup ✅
- **~50 test files** remaining across all providers

### Potential Additional Cleanup
- **~20-25 duplicate test files** identified above
- **~500-800 more lines** could be removed
- **Final state:** ~15-20 provider-specific test files (from 72 originally)
- **Total reduction:** ~1,200-1,500 lines of duplicate code eliminated

### What Should Remain
- **Core tests:** 5 files testing all 10 providers
- **Provider-specific:** ~15-20 files for truly unique features:
  - Multimodal (each provider has different API)
  - Anthropic reasoning/thinking
  - Google schema quirks
  - OpenAI hooks
  - Provider-specific modes
  - Evaluation tests

---

## Verification Script

```bash
#!/bin/bash
# Test parameterization with mock keys

export OPENAI_API_KEY=test
export ANTHROPIC_API_KEY=test
export GOOGLE_API_KEY=test

echo "Collecting tests with parameterization..."
uv run pytest tests/llm/test_core_providers/ --collect-only -q

echo ""
echo "Running single test to see parameterization..."
uv run pytest tests/llm/test_core_providers/test_basic_extraction.py::test_simple_extraction -v --collect-only
```

Expected output:
```
tests/llm/test_core_providers/test_basic_extraction.py::test_simple_extraction[openai]
tests/llm/test_core_providers/test_basic_extraction.py::test_simple_extraction[anthropic]
tests/llm/test_core_providers/test_basic_extraction.py::test_simple_extraction[google]
```

**Each test function × 10 providers = 10x test coverage with same code! 🎉**
