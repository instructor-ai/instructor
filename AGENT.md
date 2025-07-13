# AGENT.md

## Commands
- Install: `uv pip install -e ".[dev]"` or `poetry install --with dev`
- Run tests: `uv run pytest tests/`
- Run single test: `uv run pytest tests/path_to_test.py::test_name`
- Skip LLM tests: `uv run pytest tests/ -k 'not llm and not openai'`
- Type check: `uv run pyright`
- Lint: `uv run ruff check instructor examples tests`
- Format: `uv run ruff format instructor examples tests`
- Build docs: `uv run mkdocs serve` (local) or `./build_mkdocs.sh` (production)

## Architecture
- **Core**: `instructor/` - Pydantic-based structured outputs for LLMs
- **Base classes**: `Instructor` and `AsyncInstructor` in `client.py`
- **Providers**: Client files (`client_*.py`) for OpenAI, Anthropic, Gemini, Cohere, etc.
- **Factory pattern**: `from_provider()` for automatic provider detection
- **DSL**: `dsl/` directory with Partial, Iterable, Maybe, Citation extensions
- **Key modules**: `patch.py` (patching), `process_response.py` (parsing), `function_calls.py` (schemas)

## Code Style
- **Typing**: Strict type annotations, use `BaseModel` for structured outputs
- **Imports**: Standard lib → third-party → local
- **Formatting**: Ruff with Black conventions
- **Error handling**: Custom exceptions from `exceptions.py`, Pydantic validation
- **Naming**: `snake_case` functions/variables, `PascalCase` classes
- **No mocking**: Tests use real API calls
