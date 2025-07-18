# OpenAISchema Refactor Plan

## Phase 1: Create New Schema Utils

- [ ] Create `instructor/schema_utils.py` with standalone functions
  - [ ] `generate_openai_schema(model: Type[BaseModel]) -> dict[str, Any]`
  - [ ] `generate_anthropic_schema(model: Type[BaseModel]) -> dict[str, Any]`
  - [ ] `generate_gemini_schema(model: Type[BaseModel]) -> Any`
  - [ ] Add LRU cache decorators for performance
  - [ ] Import and use existing utilities (docstring_parser, etc.)

- [ ] Update `instructor/function_calls.py`
  - [ ] Import new schema utils functions
  - [ ] Update `OpenAISchema.openai_schema` to delegate to `generate_openai_schema(cls)`
  - [ ] Update `OpenAISchema.anthropic_schema` to delegate to `generate_anthropic_schema(cls)`
  - [ ] Update `OpenAISchema.gemini_schema` to delegate to `generate_gemini_schema(cls)`

- [ ] Add tests for new functions
  - [ ] Test that `generate_openai_schema(Model)` == `Model.openai_schema`
  - [ ] Test that `generate_anthropic_schema(Model)` == `Model.anthropic_schema`
  - [ ] Test that `generate_gemini_schema(Model)` == `Model.gemini_schema`

## Phase 2: Update Internal Usage (Utils Directory)

- [ ] `instructor/utils/cerebras.py` (2 calls)
  - [ ] Line 45: `"function": response_model.openai_schema,` -> `"function": generate_openai_schema(response_model),`
  - [ ] Line 50: `"function": {"name": response_model.openai_schema["name"]},` -> `"function": {"name": generate_openai_schema(response_model)["name"]},`

- [ ] `instructor/utils/writer.py` (1 call)
  - [ ] Line 62: `"function": response_model.openai_schema,` -> `"function": generate_openai_schema(response_model),`

- [ ] `instructor/utils/fireworks.py` (2 calls)
  - [ ] Line 58: `"function": response_model.openai_schema,` -> `"function": generate_openai_schema(response_model),`
  - [ ] Line 63: `"function": {"name": response_model.openai_schema["name"]},` -> `"function": {"name": generate_openai_schema(response_model)["name"]},`

- [ ] `instructor/utils/openai.py` (6 calls)
  - [ ] Line 121: `new_kwargs["functions"] = [response_model.openai_schema]` -> `new_kwargs["functions"] = [generate_openai_schema(response_model)]`
  - [ ] Line 122: `new_kwargs["function_call"] = {"name": response_model.openai_schema["name"]}` -> `new_kwargs["function_call"] = {"name": generate_openai_schema(response_model)["name"]}`
  - [ ] Line 145: `"function": response_model.openai_schema,` -> `"function": generate_openai_schema(response_model),`
  - [ ] Line 150: `"function": {"name": response_model.openai_schema["name"]},` -> `"function": {"name": generate_openai_schema(response_model)["name"]},`
  - [ ] Line 185: `"name": response_model.openai_schema["name"],` -> `"name": generate_openai_schema(response_model)["name"],`
  - [ ] Line 217: `"name": response_model.openai_schema["name"],` -> `"name": generate_openai_schema(response_model)["name"],`

- [ ] `instructor/utils/mistral.py` (1 call)
  - [ ] Line 67: `"function": response_model.openai_schema,` -> `"function": generate_openai_schema(response_model),`

- [ ] `instructor/utils/anthropic.py` (1 call)
  - [ ] Line 214: `tool_descriptions = response_model.anthropic_schema` -> `tool_descriptions = generate_anthropic_schema(response_model)`

- [ ] `instructor/utils/google.py` (1 call)
  - [ ] Line 644: `new_kwargs["tools"] = [response_model.gemini_schema]` -> `new_kwargs["tools"] = [generate_gemini_schema(response_model)]`

- [ ] `instructor/utils/core.py` (1 call)
  - [ ] Line 605: `response_model = openai_schema(response_model)` -> keep for now (decorator usage)

## Phase 3: Update Core Processing

- [ ] `instructor/process_response.py` (11 calls)
  - [ ] Add import for new schema utils
  - [ ] Replace all `response_model.openai_schema` calls with `generate_openai_schema(response_model)`
  - [ ] Replace all `response_model.anthropic_schema` calls with `generate_anthropic_schema(response_model)`

## Phase 4: Update DSL and Specialized Components

- [ ] `instructor/dsl/parallel.py` (3 calls)
  - [ ] Line 111: `{"type": "function", "function": openai_schema(model).openai_schema}` -> `{"type": "function", "function": generate_openai_schema(model)}`
  - [ ] Line 120: `return [openai_schema(model).anthropic_schema for model in the_types]` -> `return [generate_anthropic_schema(model) for model in the_types]`
  - [ ] Update imports

- [ ] `instructor/distil.py` (1 call)
  - [ ] Line 234: `openai_function_call = openai_schema(base_model).openai_schema` -> `openai_function_call = generate_openai_schema(base_model)`

## Phase 5: Update Public API

- [ ] `instructor/__init__.py`
  - [ ] Add exports for new schema utils functions
  - [ ] Keep existing exports for backward compatibility

## Phase 6: Add Deprecation Warnings

- [ ] Update `openai_schema` decorator function
  - [ ] Add deprecation warning pointing to new functions
  - [ ] Keep existing functionality

- [ ] Update `gemini_schema` property
  - [ ] Keep existing deprecation warning
  - [ ] Update message to point to new function

## Phase 7: Documentation and Examples

- [ ] Update examples to use new functions where appropriate
  - [ ] `examples/patching/pcalls.py` (4 calls)
  - [ ] `examples/gpt-engineer/refactor.py` (2 calls)
  - [ ] `examples/gpt-engineer/generate.py` (2 calls)
  - [ ] `examples/citation_with_extraction/main.py` (2 calls)
  - [ ] `examples/query_planner_execution/query_planner_execution.py` (2 calls)
  - [ ] `examples/safer_sql_example/safe_sql.py` (2 calls)

- [ ] Update documentation
  - [ ] Add new functions to API docs
  - [ ] Add migration guide
  - [ ] Update quick start examples

## Phase 8: Testing and Validation

- [ ] Run existing test suite to ensure no regressions
- [ ] Update tests that directly test schema generation
  - [ ] `tests/test_schema.py` (multiple calls)
  - [ ] `tests/test_function_calls.py` (multiple calls)
  - [ ] `tests/test_dynamic_model_creation.py` (multiple calls)
  - [ ] `tests/test_multitask.py` (2 calls)

- [ ] Add performance benchmarks comparing old vs new approach
- [ ] Verify all provider schemas remain identical

## Phase 9: Cleanup (Future)

- [ ] Remove OpenAISchema inheritance requirement (breaking change)
- [ ] Remove decorator function (breaking change)
- [ ] Move schema generation logic entirely to utils

## Notes

- **Priority**: Focus on Phase 1-4 first (core functionality)
- **Testing**: Run tests after each phase
- **Backward Compatibility**: All existing code must continue working through Phase 8
- **Performance**: LRU cache ensures no performance regression
- **Provider Coverage**: Ensure all providers (OpenAI, Anthropic, Gemini, Cerebras, Fireworks, Mistral, Writer, Cohere, VertexAI) work correctly
