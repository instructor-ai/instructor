# AGENT.md - Core Library

## Key Modules
- `client.py` - Base `Instructor`/`AsyncInstructor` classes
- `client_*.py` - Provider-specific implementations (OpenAI, Anthropic, etc.)
- `patch.py` - Core patching logic for LLM clients
- `process_response.py` - Response parsing and validation
- `function_calls.py` - Schema generation from Pydantic models
- `auto_client.py` - `from_provider()` factory pattern

## Architecture Patterns
- **Factory pattern**: `from_openai()`, `from_anthropic()` etc.
- **Patching**: Modify provider clients to add `response_model` parameter
- **Mode system**: `Mode` enum for provider capabilities (tools vs JSON)
- **Validation**: Pydantic models for structured outputs
- **Retry logic**: `retry.py` handles validation failures

## Development
- **Provider support**: Each provider needs `client_*.py` + tests in `tests/llm/`
- **Testing**: Real API calls, no mocking (see `tests/llm/test_*/conftest.py`)
- **Type safety**: Strict typing with `BaseModel` constraints
- **Error handling**: Custom exceptions in `exceptions.py`
- **CLI tools**: `cli/` directory for batch processing and utilities

## DSL Extensions
- `dsl/partial.py` - Streaming partial objects
- `dsl/iterable.py` - Lists and iterables
- `dsl/maybe.py` - Optional/nullable types
- `dsl/citation.py` - Source attribution
