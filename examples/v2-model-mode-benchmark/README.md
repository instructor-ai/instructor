# V2 model and mode benchmark

This example runs a small live structured-extraction probe across the v2
provider/mode contract. It ranks completed cells by success rate and median
request latency, and reports skipped cells when a default model, optional SDK,
or required API key is unavailable.

The default run enumerates every provider that declares v2 handlers and every
supported mode in `ProviderSpec`:

```bash
uv run python examples/v2-model-mode-benchmark/run.py --trials 3
```

Only set credentials for providers you want to run. Missing credentials produce
`skipped` rows instead of errors. Common variables include
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY`,
`XAI_API_KEY`, `MISTRAL_API_KEY`, `FIREWORKS_API_KEY`, `CEREBRAS_API_KEY`,
and `WRITER_API_KEY`.

Bedrock and Vertex AI can use ambient cloud credential chains rather than a
single API key. They are skipped by default; include `--allow-cloud-auth` when
your local cloud credentials and project/region settings are configured.

To compare selected models and modes:

```bash
uv run python examples/v2-model-mode-benchmark/run.py \
  --model openai/gpt-4o-mini \
  --model anthropic/claude-sonnet-4-6 \
  --mode TOOLS \
  --mode JSON_SCHEMA \
  --trials 5 \
  --markdown-out benchmark-results.md \
  --json-out benchmark-results.json
```

This is a smoke benchmark, not a model leaderboard. Replace or expand the
prompt and response model when measuring an application-specific workload.
