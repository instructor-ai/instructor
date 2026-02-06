# AGENT.md

## Commands
- Install: `uv pip install -e ".[dev]"` or `poetry install --with dev`
- Run tests: `uv run pytest tests/`
- Run single test: `uv run pytest tests/path_to_test.py::test_name`
- Skip LLM tests: `uv run pytest tests/ -k 'not llm and not openai'`
- Temp deps for a run: `uv run --with <pkg>[==version] <command>` (example: `uv run --with pytest-asyncio --with anthropic pytest tests/...`)
- Type check: `uv run ty check`
- Lint: `uv run ruff check instructor examples tests`
- Format: `uv run ruff format instructor examples tests`
- Build docs: `uv run mkdocs serve` (local) or `./build_mkdocs.sh` (production)
- Waiting: use `sleep <seconds>` for explicit pauses (e.g., CI waits) or to let external processes finish

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
- **Client creation**: Always use `instructor.from_provider("provider_name/model_name")` instead of provider-specific methods like `from_openai()`, `from_anthropic()`, etc.

## Pull Request (PR) Formatting

Use **Conventional Commits** formatting for PR titles. Treat the PR title as the message we would use for a squash merge commit.

### PR Title Format

Use:

`<type>(<scope>): <short summary>`

Rules:
- Keep it under ~70 characters when you can.
- Use the imperative mood (for example, “add”, “fix”, “update”).
- Do not end with a period.
- If it includes a breaking change, add `!` after the type or scope (for example, `feat(api)!:`).

Good examples:
- `fix(openai): handle empty tool_calls in streaming`
- `feat(retry): add backoff for JSON parse failures`
- `docs(agents): add conventional commit PR title guidelines`
- `test(schema): cover nested union edge cases`
- `ci(ruff): enforce formatting in pre-commit`

Common types:
- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation-only changes
- `refactor`: code change that is not a fix or feature
- `perf`: performance improvement
- `test`: add or update tests
- `build`: build system or dependency changes
- `ci`: CI pipeline changes
- `chore`: maintenance work

Suggested scopes (pick the closest match):
- Providers: `openai`, `anthropic`, `gemini`, `vertexai`, `bedrock`, `mistral`, `groq`, `writer`
- Core: `core`, `patch`, `process_response`, `function_calls`, `retry`, `dsl`
- Repo: `docs`, `examples`, `tests`, `ci`, `build`

### PR Description Guidelines

Keep PR descriptions short and easy to review:
- **What**: What changed, in 1–3 sentences.
- **Why**: Why this change is needed (link issues when possible).
- **Changes**: 3–7 bullet points with the main edits.
- **Testing**: What you ran (or why you did not run anything).

If the PR was authored by Cursor, include:
- `This PR was written by [Cursor](https://cursor.com)`
