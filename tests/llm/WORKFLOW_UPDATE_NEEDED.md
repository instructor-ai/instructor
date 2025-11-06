# GitHub Actions Workflow Update Needed

The test consolidation requires updates to `.github/workflows/test.yml` that couldn't be pushed automatically due to permission restrictions.

## Required Changes to `.github/workflows/test.yml`

### 1. Add New Core Provider Tests Job

Add this new job after the `core-tests` job:

```yaml
  # Core provider tests (unified tests across all providers)
  core-provider-tests:
    name: Core Provider Tests (All Providers)
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2
      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      - name: Set up Python
        run: uv python install 3.11
      - name: Install the project
        run: uv sync --all-extras
      - name: Run core provider tests
        run: uv run pytest tests/llm/test_core_providers/ -n auto
        env:
          INSTRUCTOR_ENV: CI
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
          COHERE_API_KEY: ${{ secrets.COHERE_API_KEY }}
          XAI_API_KEY: ${{ secrets.XAI_API_KEY }}
          MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
          CEREBRAS_API_KEY: ${{ secrets.CEREBRAS_API_KEY }}
          FIREWORKS_API_KEY: ${{ secrets.FIREWORKS_API_KEY }}
          WRITER_API_KEY: ${{ secrets.WRITER_API_KEY }}
          PERPLEXITY_API_KEY: ${{ secrets.PERPLEXITY_API_KEY }}
```

### 2. Update Provider-Specific Tests Job

Rename the `provider-tests` job to `provider-specific-tests` and update the matrix:

```yaml
  # Provider-specific tests (features unique to each provider)
  provider-specific-tests:
    name: ${{ matrix.provider.name }} Specific Tests
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        provider:
          - name: OpenAI
            env_key: OPENAI_API_KEY
            test_path: tests/llm/test_openai
          - name: Anthropic
            env_key: ANTHROPIC_API_KEY
            test_path: tests/llm/test_anthropic
          - name: Gemini
            env_key: GOOGLE_API_KEY
            test_path: tests/llm/test_gemini
          - name: Google GenAI
            env_key: GOOGLE_API_KEY
            test_path: tests/llm/test_genai
          - name: Cohere
            env_key: COHERE_API_KEY
            test_path: tests/llm/test_cohere
          - name: XAI
            env_key: XAI_API_KEY
            test_path: tests/llm/test_xai
          - name: Mistral
            env_key: MISTRAL_API_KEY
            test_path: tests/llm/test_mistral
          - name: Writer
            env_key: WRITER_API_KEY
            test_path: tests/llm/test_writer
```

Note: Removed Cerebras, Fireworks, and Perplexity from the matrix since those test directories were deleted.

## Why These Changes Are Needed

1. **New core-provider-tests job**: Runs the unified test suite in `tests/llm/test_core_providers/` against all 10 providers simultaneously

2. **Updated provider-specific-tests**: Now only runs provider-specific feature tests (like multimodal, reasoning, etc.) for providers that have unique features

3. **Deleted providers**: Cerebras, Fireworks, and Perplexity test directories were removed since their tests are now in the core test suite

## Required GitHub Secrets

Ensure these secrets are configured in the repository (tests will skip gracefully if missing):

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `COHERE_API_KEY`
- `XAI_API_KEY`
- `MISTRAL_API_KEY`
- `CEREBRAS_API_KEY`
- `FIREWORKS_API_KEY`
- `WRITER_API_KEY`
- `PERPLEXITY_API_KEY`
