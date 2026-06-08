# Provider Architecture and Parity Inventory

This page records the provider architecture introduced by the
`provider-owned-clients` change and the compatibility decisions verified on
June 7, 2026. The measured baseline is commit `0212d8833cba`, immediately
before the shared native-client factory migration.

The implementation, documentation, and deterministic contract tests are split
into three reviewable changes. The documentation and tests changes both target
the same core implementation commit; the test evidence and measurements below
are supplied by the companion tests change rather than duplicated here.

## Architecture decisions

- `ProviderSpec` is the provider control plane. It declares aliases, canonical
  provider identity, supported and legacy modes, handler modules, builders,
  capabilities, dependency errors, and the native client contract.
- `ClientSpec` is an internal declarative contract for native sync and async SDK
  types, create and stream method paths, model fallback behavior, and legacy
  validation precedence. It is not a public export.
- `create_instructor()` is the single implementation of native client
  validation, mode normalization, sync/async selection, method resolution,
  stream switching, patching, model defaults, and wrapper construction.
- The mode registry remains the single runtime lookup for request, re-ask,
  response, sync stream, async stream, message, and template handlers. Legacy
  provider ownership and normalization are derived from `ProviderSpec`.
- `retry_sync_v2()` and `retry_async_v2()` own validation retries, re-asks,
  usage accumulation, hook emission, error propagation, and final response
  attachment for every retained provider.
- Provider modules retain only SDK construction and behavior that cannot be
  expressed by the common contract. OpenAI Responses, xAI, and LiteLLM remain
  explicit compatibility exceptions.

## Public export inventory

Every name in `instructor.__all__` remains available. Optional factories remain
conditional on their SDK being importable, matching the previous behavior.

| Export | Result |
| --- | --- |
| `Instructor` | Preserved |
| `AsyncInstructor` | Preserved |
| `Image` | Preserved |
| `Audio` | Preserved |
| `from_openai` | Preserved |
| `from_anyscale` | Preserved |
| `from_together` | Preserved |
| `from_databricks` | Preserved |
| `from_deepseek` | Preserved |
| `from_openrouter` | Preserved |
| `from_litellm` | Preserved |
| `from_vertexai` | Preserved; deprecated provider guidance is unchanged |
| `from_provider` | Preserved |
| `Provider` | Preserved |
| `ResponseSchema` | Preserved |
| `response_schema` | Preserved |
| `OpenAISchema` | Preserved |
| `CitationMixin` | Preserved |
| `IterableModel` | Preserved |
| `Maybe` | Preserved |
| `Partial` | Preserved |
| `openai_schema` | Preserved |
| `generate_openai_schema` | Preserved |
| `generate_anthropic_schema` | Preserved |
| `generate_gemini_schema` | Preserved |
| `Mode` | Preserved |
| `patch` | Preserved |
| `apatch` | Preserved |
| `FinetuneFormat` | Preserved |
| `Instructions` | Preserved |
| `BatchProcessor` | Preserved |
| `BatchRequest` | Preserved |
| `BatchJob` | Preserved |
| `llm_validator` | Preserved |
| `openai_moderation` | Preserved |
| `hooks` | Preserved |
| `v2` | Preserved |
| `from_anthropic` | Preserved as an optional export |
| `from_gemini` | Preserved as an optional export |
| `from_fireworks` | Preserved as an optional export |
| `from_cerebras` | Preserved as an optional export |
| `from_groq` | Preserved as an optional export |
| `from_mistral` | Preserved as an optional export |
| `from_cohere` | Preserved as an optional export |
| `from_bedrock` | Preserved as an optional export |
| `from_writer` | Preserved as an optional export |
| `from_xai` | Preserved as an optional export |
| `from_perplexity` | Preserved as an optional export |
| `from_genai` | Preserved as an optional export |

`instructor.__version__` also remains available. No public export was removed.

## Provider and mode inventory

The table lists canonical modes after normalization. Existing provider-specific
mode enum values remain accepted as deprecated aliases where declared by the
provider specification.

| Provider or alias | Factory | Canonical modes | Result |
| --- | --- | --- | --- |
| OpenAI | `from_openai` | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS`, `RESPONSES_TOOLS` | Preserved |
| Anyscale | `from_anyscale` | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | Preserved through OpenAI-compatible handlers |
| Together | `from_together` | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | Preserved through OpenAI-compatible handlers |
| Databricks | `from_databricks` | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | Preserved through OpenAI-compatible handlers |
| DeepSeek | `from_deepseek` | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | Preserved through OpenAI-compatible handlers |
| OpenRouter | `from_openrouter` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | Preserved |
| Anthropic | `from_anthropic` | `TOOLS`, `JSON`, `JSON_SCHEMA`, `PARALLEL_TOOLS` | Preserved, including beta client routing |
| Google GenAI (`google/...`) | `from_genai` | `TOOLS`, `JSON` | Preserved |
| `generative-ai` | `from_provider` alias | Canonicalizes to Google GenAI | Preserved deprecated alias |
| Gemini | `from_gemini` | `TOOLS`, `MD_JSON` | Preserved deprecated SDK integration |
| Cohere | `from_cohere` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON` | Preserved, including V1/V2 and split stream methods |
| Perplexity | `from_perplexity` | `MD_JSON` | Preserved |
| xAI | `from_xai` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | Preserved custom adapter |
| Groq | `from_groq` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON` | Preserved |
| Mistral | `from_mistral` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON` | Preserved, including split stream methods |
| Fireworks | `from_fireworks` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON` | Preserved |
| Cerebras | `from_cerebras` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | Preserved |
| Writer | `from_writer` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON` | Preserved |
| Bedrock | `from_bedrock` | `TOOLS`, `MD_JSON` | Preserved, including async wrapping of the sync SDK method |
| Vertex AI | `from_vertexai` | `TOOLS`, `MD_JSON`, `PARALLEL_TOOLS` | Preserved deprecated provider path |
| Azure OpenAI | `from_provider` alias | Canonicalizes to OpenAI | Preserved compatibility alias |
| Ollama | `from_provider` alias | Canonicalizes to OpenAI | Preserved compatibility alias |
| LiteLLM | `from_litellm` | Callable wrapper modes | Preserved custom adapter |

Unsupported provider/mode combinations continue to raise `ModeError`; they are
not advertised as capabilities by `ProviderSpec`.

## Workflow parity

| Workflow | Result | Deterministic evidence |
| --- | --- | --- |
| Sync structured extraction | Preserved | Shared provider/client, handler, patch, and retry suites |
| Async structured extraction | Preserved | Shared wrapper selection and async retry suites |
| Pydantic response validation | Preserved | Shared response parser and retry runtime suites |
| Full streaming | Preserved | Provider capability runtime contracts |
| Async streaming | Preserved | Async extractor contract for every advertised stream mode |
| `Partial[T]` streaming | Preserved | DSL and provider capability suites |
| `Iterable[T]` responses and streaming | Preserved | DSL, response-list, and provider capability suites |
| Retries and re-asks | Preserved | Central sync/async retry runtime suites |
| `completion:kwargs` and `completion:response` hooks | Preserved | Retry runtime suites |
| `completion:error` and `completion:last_attempt` hooks | Intentionally changed to match the documented contract | Raw and structured failure hook tests |
| Usage accumulation and reporting | Preserved | Shared retry usage paths and provider usage tests |
| `Maybe`, citations, schemas, multimodal DSL | Preserved | Public typing and deterministic DSL suites |
| Automatic provider detection | Preserved | `from_provider` and provider URL detection suites |
| Provider-specific factory imports | Preserved | Lazy public import and legacy compatibility suites |
| Documented Mistral async workflow | Preserved and corrected | Uses `async_client=True` explicitly |
| Documented Bedrock async workflow | Preserved and corrected | Uses `async_client=True` explicitly |

## Intentional changes and removals

| Item | Result | Evidence and migration |
| --- | --- | --- |
| Provider-local sync/async validation, patching, stream switching, and wrapper construction | Removed as obsolete duplication | Replaced by `ClientSpec` and `create_instructor()`; public factory signatures are unchanged |
| Repeated Bedrock, Cerebras, Fireworks, Groq, Mistral, and Writer client suites | Removed as meaningless duplication | Replaced by the manifest-driven shared client contract; provider-specific handler tests remain |
| Bedrock `_async` documentation | Removed as broken or obsolete | `_async` was not a supported selector; pass `async_client=True` to `from_provider()` or use the async factory overload |
| Claim that provider-specific helpers were removed | Removed as incorrect documentation | Native-client helpers remain public; prefer `from_provider()` for model-string construction |
| Vertex AI deterministic test without the optional SDK | Intentionally changed | The test now skips when `vertexai` is unavailable instead of failing during patch setup |
| Final failure hook behavior | Intentionally changed to restore documented behavior | `completion:error` and `completion:last_attempt` now receive attempt metadata; existing hook signatures remain backward-compatible |

No provider, working mode, DSL type, retry path, stream workflow, hook name, or
documented supported factory was removed.

## Measurements

Measurements compare the working tree with `0212d8833cba` using the same
fully populated `.venv`, with proxy variables unset. Generated files and
dependency locks are excluded. Production LOC counts nonblank, non-comment
implementation-body lines and excludes overloads and docstrings; test LOC is
physical file length for the replaced contract slice.

| Slice | Baseline | Current | Reduction |
| --- | ---: | ---: | ---: |
| Executable body LOC in the 11 migrated `from_*` implementations | 589 LOC | 150 LOC | 74.5% |
| Provider client contract tests replaced by `test_client_unified.py` | 869 LOC | 260 LOC | 70.1% |

The selected deterministic parity command is:

```bash
python -m pytest \
  tests/v2/test_provider_specs.py \
  tests/v2/test_client_unified.py \
  tests/v2/test_auto_client_deterministic.py \
  tests/providers/test_auto_client.py -q
```

For the selected parity command, three baseline runs from an isolated archive
had a 16.68-second median with 513 passing and 46 skipped. Three working-tree
runs had a 16.28-second median with 101 passing and 27 skipped, a 2.4% runtime
reduction. The complete deterministic `tests/v2 tests/providers` suite improved
from a 34.76-second three-run median (1,644 passing, 199 skipped) to a
32.88-second median (1,279 passing, 170 skipped), a 5.4% reduction. Optional SDK
imports and pytest startup dominate both measurements, so the 75% timing target
remains unmet even though repeated test cases and provider implementations were
removed.

Browser automation verified the rendered `from_provider`, Mistral, Bedrock, and
provider parity pages on the local MkDocs site. The parity inventory rendered
all five tables without console errors.
