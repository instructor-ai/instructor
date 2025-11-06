# Parameterization Analysis & Provider-Specific Test Identification

## 1. Parameterization Status

### How It Works

The parameterization uses pytest's `pytest_generate_tests` hook in `conftest.py`:

```python
# tests/llm/test_core_providers/conftest.py
from shared_config import pytest_generate_tests  # noqa: F401
```

This imports the hook from `shared_config.py`:

```python
def pytest_generate_tests(metafunc):
    if "provider_config" in metafunc.fixturenames:
        available = get_available_providers()
        if not available:
            pytest.skip("No providers available (missing API keys or packages)")

        ids = [model.split("/")[0] for model, _ in available]
        metafunc.parametrize("provider_config", available, ids=ids)
```

### Current Status

**✅ Parameterization IS Working**

When API keys are available, each test function with `provider_config` parameter will run once per available provider.

Example:
```python
def test_simple_extraction(provider_config):
    model, mode = provider_config  # e.g., ("openai/gpt-5-nano", Mode.TOOLS)
    client = instructor.from_provider(model, mode=mode)
    # test code...
```

This will generate:
- `test_simple_extraction[openai]` - if OPENAI_API_KEY set
- `test_simple_extraction[anthropic]` - if ANTHROPIC_API_KEY set
- `test_simple_extraction[google]` - if GOOGLE_API_KEY set
- ... for all 10 configured providers

**When No API Keys Are Set:**
- Tests are skipped with message: "No providers available (missing API keys or packages)"
- This is CORRECT behavior - tests gracefully skip instead of failing

## 2. Provider-Specific Test Analysis

### Truly Provider-Specific Tests (Keep Separate)

#### test_openai/
1. **test_hooks.py** ✅ PROVIDER-SPECIFIC
   - Tests OpenAI's hook system (`instructor.hooks`)
   - Tests: `test_on_method_str`, `test_on_method_enum`, etc.
   - Uses `instructor.hooks.HookName` enum
   - NOT generic - specific to instructor's OpenAI hook implementation

2. **test_multimodal.py** ✅ PROVIDER-SPECIFIC
   - Tests OpenAI's vision/multimodal API
   - Uses OpenAI-specific image URL format
   - Different from Anthropic/Google multimodal APIs

3. **test_validation_context.py** ✅ PROVIDER-SPECIFIC
   - Tests validation context passing in OpenAI
   - Provider-specific validation behavior

4. **test_attr.py** - REVIEW NEEDED
   - May be generic attribute access tests

5. **test_modes.py** - REVIEW NEEDED
   - May test generic mode switching

6. **test_parallel.py** - **COULD BE GENERIC**
   - Parallel requests likely work same across providers

7. **test_retries.py** - **COULD BE GENERIC**
   - Retry logic should be provider-agnostic

8. **test_stream.py** - **COULD BE GENERIC**
   - Streaming should work same across providers

9. **test_validators.py** - **COULD BE GENERIC**
   - Pydantic validators are provider-agnostic

#### test_anthropic/
1. **test_reasoning.py** ✅ PROVIDER-SPECIFIC
   - Tests Anthropic's `thinking` parameter
   - Uses `Mode.ANTHROPIC_REASONING_TOOLS`
   - Anthropic-only feature

2. **test_system.py** ✅ PROVIDER-SPECIFIC
   - Tests Anthropic's system prompt handling
   - Anthropic has unique system message requirements

3. **test_multimodal.py** ✅ PROVIDER-SPECIFIC
   - Tests Anthropic's multimodal API (different from OpenAI)
   - Uses Anthropic-specific image formats

4. **test_parallel.py** - **COULD BE GENERIC**
   - Parallel requests likely work same across providers

5. **test_stream.py** - **ALREADY IN CORE**
   - Should be removed or kept for Anthropic-specific streaming features

#### test_genai/
1. **test_schema_conversion.py** ✅ PROVIDER-SPECIFIC
   - Tests Google's schema conversion logic
   - Google has unique schema requirements

2. **test_format.py** ✅ PROVIDER-SPECIFIC
   - Tests Google-specific format handling

3. **test_decimal.py** ✅ PROVIDER-SPECIFIC
   - Tests Google's decimal number handling quirks

4. **test_invalid_schema.py** ✅ PROVIDER-SPECIFIC
   - Tests Google's schema validation errors

5. **test_multimodal.py** ✅ PROVIDER-SPECIFIC
   - Tests Google's multimodal API (different from OpenAI/Anthropic)

6. **test_utils.py** ✅ PROVIDER-SPECIFIC
   - Tests Google-specific utilities

7. **test_basics.py** - **COULD BE GENERIC**
8. **test_simple.py** - **COULD BE GENERIC**
9. **test_stream.py** - **ALREADY IN CORE**
10. **test_response_model_none.py** - **ALREADY IN CORE**

#### test_gemini/
1. **test_list_content.py** ✅ PROVIDER-SPECIFIC
   - Tests Gemini's list content formatting

2. **test_multimodal_content.py** ✅ PROVIDER-SPECIFIC
   - Tests Gemini's multimodal content structure

3. **evals/** ✅ PROVIDER-SPECIFIC
   - Evaluation tests for Gemini model capabilities

4. **test_patch.py** - **COULD BE GENERIC**
5. **test_retries.py** - **ALREADY IN CORE**
6. **test_simple_types.py** - **COULD BE GENERIC**
7. **test_stream.py** - **ALREADY IN CORE**

#### test_cohere/
1. **test_json_schema.py** ✅ PROVIDER-SPECIFIC
   - Tests Cohere's JSON schema mode
   - `Mode.COHERE_JSON_SCHEMA` is Cohere-specific

2. **test_none_response.py** - **ALREADY IN CORE**
3. **test_retries.py** - **ALREADY IN CORE**

#### test_xai/
1. **test_raw_response.py** ✅ PROVIDER-SPECIFIC (maybe)
   - Tests raw response handling
   - Needs review - could be generic

2. **test_basics.py** - **ALREADY IN CORE**
3. **test_stream.py** - **ALREADY IN CORE**

#### test_mistral/
1. **test_multimodal.py** ✅ PROVIDER-SPECIFIC
   - Tests Mistral's multimodal API

2. **test_modes.py** - **ALREADY IN CORE** (modes should work same)
3. **test_retries.py** - **ALREADY IN CORE**
4. **test_stream.py** - **ALREADY IN CORE**

#### test_writer/
1. **evals/** ✅ PROVIDER-SPECIFIC
   - Evaluation tests for Writer model capabilities

2. **test_format_common_models.py** - REVIEW NEEDED
3. **test_format_difficult_models.py** - REVIEW NEEDED
4. **test_retries.py** - **ALREADY IN CORE**
5. **test_streaming.py** - **ALREADY IN CORE**

## 3. Recommendations for Further Cleanup

### Can Be Deleted (Already in Core)
- `test_openai/test_stream.py` - duplicate of core streaming tests
- `test_openai/test_retries.py` - duplicate of core retry tests
- `test_openai/test_parallel.py` - parallel should work same across providers
- `test_anthropic/test_stream.py` - duplicate of core
- `test_anthropic/test_parallel.py` - duplicate
- `test_genai/test_basics.py` - duplicate
- `test_genai/test_simple.py` - duplicate
- `test_genai/test_stream.py` - duplicate
- `test_genai/test_response_model_none.py` - duplicate
- `test_gemini/test_retries.py` - duplicate
- `test_gemini/test_stream.py` - duplicate
- `test_cohere/test_none_response.py` - duplicate
- `test_cohere/test_retries.py` - duplicate
- `test_xai/test_basics.py` - duplicate
- `test_xai/test_stream.py` - duplicate
- `test_mistral/test_modes.py` - duplicate
- `test_mistral/test_retries.py` - duplicate
- `test_mistral/test_stream.py` - duplicate
- `test_writer/test_retries.py` - duplicate
- `test_writer/test_streaming.py` - duplicate

### Keep (Truly Provider-Specific)
- All **multimodal** tests (each provider has different API)
- All **reasoning/thinking** tests (Anthropic-only)
- All **schema conversion/format** tests (provider-specific quirks)
- All **hooks** tests (OpenAI-specific feature)
- All **system prompt** tests (Anthropic-specific)
- All **evals/** directories (model capability tests)
- Provider-specific mode tests (JSON_SCHEMA for Cohere, etc.)

### Needs Review
- `test_openai/test_attr.py`
- `test_openai/test_modes.py`
- `test_openai/test_validators.py`
- `test_gemini/test_patch.py`
- `test_gemini/test_simple_types.py`
- `test_xai/test_raw_response.py`
- `test_writer/test_format_*.py`

## 4. Estimated Additional Cleanup

If we remove the duplicate tests identified above:
- **~20-25 more test files** could be deleted
- **~500-800 more lines** of duplicate code removed
- Final state: **~10-15 provider-specific test files** total (from 72 files originally)

## 5. Verification Commands

To verify parameterization is working (with API keys set):

```bash
# Collect tests without running them
uv run pytest tests/llm/test_core_providers/test_basic_extraction.py --collect-only

# Should show:
# test_simple_extraction[openai]
# test_simple_extraction[anthropic]
# test_simple_extraction[google]
# ... etc for each available provider
```

To run with mock API keys for testing:

```bash
export OPENAI_API_KEY=test
export ANTHROPIC_API_KEY=test
export GOOGLE_API_KEY=test

uv run pytest tests/llm/test_core_providers/ --collect-only
# Should show N tests per provider
```
