# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Instructor Development Guide

## Commands
- Install deps: `uv pip install -e ".[dev,anthropic]"` or `poetry install --with dev,anthropic`
- Run tests: `pytest tests/`
- Run specific test: `pytest tests/path_to_test.py::test_name`
- Skip LLM tests: `pytest tests/ -k 'not llm and not openai'`
- Type check: `pyright`
- Lint: `ruff check instructor examples tests`
- Format: `ruff format instructor examples tests`
- Generate coverage: `coverage run -m pytest tests/ -k "not docs"` then `coverage report`
- Build documentation: `mkdocs serve` (for local preview) or `./build_mkdocs.sh` (for production)

## Code Style Guidelines
- **Typing**: Use strict typing with annotations for all functions and variables
- **Imports**: Standard lib → third-party → local imports
- **Formatting**: Follow Black's formatting conventions (enforced by Ruff)
- **Models**: Define structured outputs as Pydantic BaseModel subclasses
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use custom exceptions from exceptions.py, validate with Pydantic
- **Comments**: Docstrings for public functions, inline comments for complex logic
- **Commits**: Follow conventional commits format (type(scope): description)

## Core Architecture
- **Base Classes**: `Instructor` and `AsyncInstructor` in client.py are the foundation
- **Factory Pattern**: Provider-specific factory functions (`from_openai`, `from_anthropic`, etc.)
- **Unified Access**: `from_provider()` function in auto_client.py for automatic provider detection
- **Mode System**: `Mode` enum categorizes different provider capabilities (tools vs JSON output)
- **Patching Mechanism**: Uses Python's dynamic nature to patch provider clients for structured outputs
- **Response Processing**: Transforms raw API responses into validated Pydantic models
- **DSL Components**: Special types like Partial, Iterable, Maybe extend the core functionality

## Provider Architecture
- **Supported Providers**: OpenAI, Anthropic, Gemini, Cohere, Mistral, Groq, VertexAI, Fireworks, Cerebras, Writer, Databricks, Anyscale, Together, LiteLLM
- **Provider Implementation**: Each provider has a dedicated client file (e.g., `client_anthropic.py`) with factory functions
- **Modes**: Different providers support specific modes (`Mode` enum): `ANTHROPIC_TOOLS`, `GEMINI_JSON`, etc.
- **Common Pattern**: Factory functions (e.g., `from_anthropic`) take a native client and return patched `Instructor` instances
- **Provider Testing**: Tests in `tests/llm/` directory, define Pydantic models, make API calls, verify structured outputs
- **Provider Detection**: `get_provider` function analyzes base URL to detect which provider is being used

## Key Components
- **process_response.py**: Handles parsing and converting LLM outputs to Pydantic models
- **patch.py**: Contains the core patching logic for modifying provider clients
- **function_calls.py**: Handles generating function/tool schemas from Pydantic models
- **hooks.py**: Provides event hooks for intercepting various stages of the LLM request/response cycle
- **dsl/**: Domain-specific language extensions for specialized model types
- **retry.py**: Implements retry logic for handling validation failures
- **validators.py**: Custom validation mechanisms for structured outputs

## Testing Guidelines
- Tests are organized by provider under `tests/llm/`
- Each provider has its own conftest.py with fixtures
- Standard tests cover: basic extraction, streaming, validation, retries
- Evaluation tests in `tests/llm/test_provider/evals/` assess model capabilities
- Use parametrized tests when testing similar functionality across variants
- Mock API calls when possible to reduce costs during development

## Documentation Guidelines
- Every provider needs documentation in `docs/integrations/` following standard format
- Provider docs should include: installation, basic example, modes supported, special features
- When adding a new provider, update `mkdocs.yml` navigation and redirects
- Example code should include complete imports and environment setup
- Tutorials should progress from simple to complex concepts
- New features should include conceptual explanation in `docs/concepts/`

The library enables structured LLM outputs using Pydantic models across multiple providers with type safety.