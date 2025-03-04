# Instructor Development Guide

## Commands
- Install deps: `poetry install --with dev,anthropic`
- Run tests: `poetry run pytest tests/`
- Run specific test: `poetry run pytest tests/path_to_test.py::test_name`
- Skip LLM tests: `poetry run pytest tests/ -k 'not llm and not openai'`
- Type check: `poetry run pyright`
- Lint: `ruff check instructor examples tests`
- Format: `black instructor examples tests`
- Generate coverage: `poetry run coverage run -m pytest tests/ -k "not docs"` then `poetry run coverage report`

## Code Style Guidelines
- **Typing**: Use strict typing with annotations for all functions and variables
- **Imports**: Standard lib → third-party → local imports
- **Formatting**: Follow Black's formatting conventions
- **Models**: Define structured outputs as Pydantic BaseModel subclasses
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use custom exceptions from exceptions.py, validate with Pydantic
- **Comments**: Docstrings for public functions, inline comments for complex logic

## Provider Architecture
- **Supported Providers**: OpenAI, Anthropic, Gemini, Cohere, Mistral, Groq, VertexAI, Fireworks, Cerebras, Writer, Databricks, Anyscale, Together, LiteLLM
- **Provider Implementation**: Each provider has a dedicated client file (e.g., `client_anthropic.py`) with factory functions
- **Modes**: Different providers support specific modes (`Mode` enum): `ANTHROPIC_TOOLS`, `GEMINI_JSON`, etc.
- **Common Pattern**: Factory functions (e.g., `from_anthropic`) take a native client and return patched `Instructor` instances
- **Provider Testing**: Tests in `tests/llm/` directory, define Pydantic models, make API calls, verify structured outputs
- **Provider Detection**: `get_provider` function analyzes base URL to detect which provider is being used

## Documentation Guidelines
- Every provider needs documentation in `docs/integrations/` following standard format
- Provider docs should include: installation, basic example, modes supported, special features
- When adding a new provider, update `mkdocs.yml` navigation and redirects
- Example code should include complete imports and environment setup
- Tutorials should progress from simple to complex concepts
- New features should include conceptual explanation in `docs/concepts/`
    

Library enables structured LLM outputs using Pydantic models across multiple providers with type safety.

