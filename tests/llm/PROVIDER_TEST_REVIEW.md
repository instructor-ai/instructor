# Provider Test Review - Consolidation Analysis

## Objective
Identify which provider tests can be consolidated into `test_core_providers/` and which should remain provider-specific.

## Analysis by Provider

### ✅ test_openai (13 test files)
**Test files:**
- test_attr.py
- test_hooks.py - **PROVIDER-SPECIFIC** (OpenAI hooks)
- test_modes.py
- test_multimodal.py - **PROVIDER-SPECIFIC** (OpenAI multimodal API)
- test_multitask.py
- test_openai.py
- test_parallel.py
- test_patch.py
- test_retries.py
- test_simple_types.py
- test_stream.py
- test_validation_context.py - **PROVIDER-SPECIFIC**
- test_validators.py

**Uses:** `from_openai()` mostly
**Recommendation:**
- Keep: hooks, multimodal, validation_context (provider-specific)
- Can delete: basic extraction, streaming, retries (now in core)

---

### ✅ test_anthropic (5 test files)
**Test files:**
- test_multimodal.py - **PROVIDER-SPECIFIC** (Anthropic multimodal API)
- test_parallel.py
- test_reasoning.py - **PROVIDER-SPECIFIC** (extended thinking)
- test_stream.py
- test_system.py - **PROVIDER-SPECIFIC** (system prompt handling)

**Uses:** `from_provider()` ✅
**Recommendation:**
- Keep: multimodal, reasoning, system (provider-specific features)
- Can delete: parallel, stream (now in core)

---

### ✅ test_genai (10 test files)
**Test files:**
- test_basics.py
- test_decimal.py - **PROVIDER-SPECIFIC** (decimal handling)
- test_format.py - **PROVIDER-SPECIFIC** (format handling)
- test_invalid_schema.py - **PROVIDER-SPECIFIC** (schema validation)
- test_multimodal.py - **PROVIDER-SPECIFIC** (Google multimodal API)
- test_response_model_none.py
- test_schema_conversion.py - **PROVIDER-SPECIFIC** (schema conversion)
- test_simple.py
- test_stream.py
- test_utils.py - **PROVIDER-SPECIFIC** (utilities)

**Uses:** `from_provider()` ✅
**Recommendation:**
- Keep: decimal, format, invalid_schema, multimodal, schema_conversion, utils
- Can delete: basics, simple, stream, response_model_none (now in core)

---

### ✅ test_gemini (6 test files + evals/)
**Test files:**
- test_list_content.py - **PROVIDER-SPECIFIC** (content format)
- test_multimodal_content.py - **PROVIDER-SPECIFIC** (multimodal)
- test_patch.py
- test_retries.py
- test_simple_types.py
- test_stream.py
- evals/ - **KEEP** (evaluation tests)

**Uses:** `from_provider()` ✅
**Recommendation:**
- Keep: list_content, multimodal_content, evals
- Can delete: patch, retries, simple_types, stream (now in core)

---

### ✅ test_cohere (3 test files)
**Test files:**
- test_json_schema.py - **PROVIDER-SPECIFIC** (JSON schema mode)
- test_none_response.py
- test_retries.py

**Uses:** `from_provider()` ✅
**Recommendation:**
- Keep: json_schema (provider-specific mode)
- Can delete: none_response, retries (now in core)

---

### ✅ test_xai (3 test files)
**Test files:**
- test_basics.py
- test_raw_response.py - **MAYBE KEEP** (raw response testing)
- test_stream.py

**Uses:** `from_provider()` ✅
**Recommendation:**
- Keep: raw_response (if provider-specific behavior)
- Can delete: basics, stream (now in core)

---

### ⚠️ test_mistral (4 test files)
**Test files:**
- test_modes.py
- test_multimodal.py - **PROVIDER-SPECIFIC** (Mistral multimodal)
- test_retries.py
- test_stream.py

**Uses:** `from_mistral()` ❌
**Recommendation:**
- ADD to core providers
- Keep: multimodal
- Can delete: modes, retries, stream after migration

---

### ⚠️ test_cerebras (1 test file)
**Test files:**
- modes.py (actually contains tests)

**Uses:** `from_cerebras()` ❌
**Recommendation:**
- ADD to core providers
- Tests are generic, can all go to core after migration

---

### ⚠️ test_fireworks (3 test files)
**Test files:**
- test_format.py
- test_simple.py
- test_stream.py

**Uses:** `from_fireworks()` ❌
**Recommendation:**
- ADD to core providers
- All tests are generic

---

### ⚠️ test_writer (4 test files + evals/)
**Test files:**
- test_format_common_models.py
- test_format_difficult_models.py
- test_retries.py
- test_streaming.py
- evals/ - **KEEP**

**Uses:** `from_writer()` ❌
**Recommendation:**
- ADD to core providers
- Keep: evals/
- Can delete: all test files after migration

---

### ⚠️ test_perplexity (1 test file)
**Test files:**
- test_modes.py

**Uses:** Unknown
**Recommendation:**
- ADD to core providers
- Test is generic

---

### ⚠️ test_bedrock (unknown)
**Recommendation:**
- Review separately (AWS complexity)

---

### ⚠️ test_vertexai (unknown)
**Recommendation:**
- Review separately (may be deprecated in favor of test_genai)

---

## Summary

### Can Add to Core (Need Migration)
- ✅ Mistral - change from_mistral() to from_provider()
- ✅ Cerebras - change from_cerebras() to from_provider()
- ✅ Fireworks - change from_fireworks() to from_provider()
- ✅ Writer - change from_writer() to from_provider()
- ✅ Perplexity - change from_perplexity() to from_provider()

### Provider-Specific to Keep
- **OpenAI:** hooks, multimodal, validation_context
- **Anthropic:** multimodal, reasoning, system
- **Google (genai):** decimal, format, invalid_schema, multimodal, schema_conversion, utils
- **Gemini:** list_content, multimodal_content, evals/
- **Cohere:** json_schema
- **xAI:** raw_response (maybe)
- **Mistral:** multimodal
- **Writer:** evals/

### Can Delete After Consolidation
- test_openai: attr, modes, multitask, openai, parallel, patch, retries, simple_types, stream
- test_anthropic: parallel, stream
- test_genai: basics, simple, stream, response_model_none
- test_gemini: patch, retries, simple_types, stream
- test_cohere: none_response, retries
- test_xai: basics, stream
- test_mistral: modes, retries, stream (after migration)
- test_cerebras: modes.py (after migration)
- test_fireworks: all (after migration)
- test_writer: all except evals/ (after migration)
- test_perplexity: all (after migration)

## Directories to Completely Remove
After migration, these can be deleted entirely:
- ❌ test_cerebras (move to core)
- ❌ test_fireworks (move to core)
- ❌ test_perplexity (move to core)

## Estimated Impact
- **Before:** ~72 test files across 14 provider directories
- **After:** ~25-30 provider-specific test files + shared core tests
- **Reduction:** ~40-50 test files eliminated (deduplicated)
