# AGENT.md

## Commands

- Install: `uv sync --all-extras --group dev` or `uv pip install -e ".[dev]"` or `poetry install --with dev`
- Run tests: `uv run pytest tests/`
- Run single test: `uv run pytest tests/path_to_test.py::test_name`
- Skip LLM tests: `uv run pytest tests/ -k 'not llm and not openai'`
- Type check: `uv run ty check`
- Lint: `uv run ruff check instructor examples tests`
- Format: `uv run ruff format instructor examples tests`
- Build docs: `uv run mkdocs serve` (local) or `./build_mkdocs.sh` (production)
- Waiting: use `sleep <seconds>` for explicit pauses (e.g., CI waits) or to let external processes finish

## Documentation Examples and Doc Tests

All code examples in documentation must be executable and pass doc tests. The `pytest-examples` plugin validates that examples run correctly and match expected output.

### Doc Test Requirements

- All code examples in documentation must be executable
- Examples must pass when run via pytest doc tests
- Examples should include print statements with expected output (using `#>` prefix)

### Writing Doc Examples

- **Self-contained blocks**: Each code block runs isolated, so shared variables like `client` and `logger` must be defined within each block or skipped. Making all blocks self-contained is essential to avoid test failures.
- **Valid Python syntax**: Invalid Python code (like ellipsis `...` after keyword arguments) causes syntax errors and must be fixed by removing or replacing placeholders.
- **Skip problematic examples**: If an example cannot be made executable, consider using skip markers (verify pytest-examples support) or exclude the file from testing (see `test_examples.py` exclusions).

### Running Doc Tests

- **Standard check**: `uv run pytest tests/docs/` (lints and runs examples)
- **Update mode**: `uv run pytest tests/docs/ --update-examples` (formats code and updates expected output/logs)
  - **Warning**: `--update-examples` modifies files in place

### What Doc Tests Do

- Format code examples using ruff
- Run examples to verify they execute correctly
- Check that printed output matches expected results
- Update examples in-place when using `--update-examples`

### Doc Test Files

Doc test files are located in `tests/docs/`:

- **test_examples.py**: Tests examples in `docs/examples/*.md` (formats and runs/updates print output)
- **test_concepts.py**: Tests examples in `docs/concepts/` (formats, runs, and updates print output)
- **test_docs.py**: Tests examples in `README.md` and `docs/index.md` (formats only, no execution)
- **test_posts.py**: Tests examples in `docs/blog/posts/` (formats and runs/updates print output)

Always run doc tests before submitting documentation changes to ensure examples remain executable and up-to-date.

## Architecture

- **Core**: `instructor/` - Pydantic-based structured outputs for LLMs
- **Base classes**: `Instructor` and `AsyncInstructor` in `core/client.py`
- **Providers**: Provider implementations in `providers/` directory (v1) and `v2/providers/` directory (v2)
  - Each provider has a `client.py` with factory functions (e.g., `from_openai`, `from_anthropic`)
  - V2 providers also have `handlers.py` for mode-specific response handling
- **Factory pattern**: `from_provider()` in `auto_client.py` for automatic provider detection (recommended)
- **DSL**: `dsl/` directory with Partial, Iterable, Maybe, Citation extensions
- **Key modules**: `patch.py` (patching), `process_response.py` (parsing), `function_calls.py` (schemas)

## Code Style

- **Typing**: Strict type annotations, use `BaseModel` for structured outputs
- **Imports**: Standard lib → third-party → local
- **Formatting**: Ruff with Black conventions
- **Error handling**: Custom exceptions from `exceptions.py`, Pydantic validation
- **Naming**: `snake_case` functions/variables, `PascalCase` classes
- **Testing**: Most tests use real API calls; unit tests for handlers may use mocks for isolated testing
- **Client creation**: Prefer `instructor.from_provider("provider_name/model_name")` for new code; provider-specific methods like `from_openai()`, `from_anthropic()` are still available for direct client control

## Pull Request (PR) Formatting

Use **Conventional Commits** formatting for PR titles. Treat the PR title as the message we would use for a squash merge commit.

### PR Title Format

Use:

`<type>(<scope>): <short summary>`

Rules:

- Keep it under ~70 characters when you can.
- Use the imperative mood (for example, "add", "fix", "update").
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
