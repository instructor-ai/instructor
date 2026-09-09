# Provider support and test evidence

Inventory based on `origin/main` at `6969bb68` (2026-09-09 UTC), plus the local
Responses contracts introduced with this document. This is a test map, **not a
provider certification**. No live provider calls were executed for this change.
A supported mode, a collected test, a passed local SDK test, and an executed live
test are different kinds of evidence.

## Implemented mode inventory

The table below transcribes [ProviderSpec](../../instructor/v2/core/provider_specs.py).
Names are normalized `Mode` members, not a claim that every upstream model accepts
that API feature. Legacy names map through each spec's `legacy_modes`; for example,
`GENAI_STRUCTURED_OUTPUTS` maps to `JSON`, not `JSON_SCHEMA`. Alias rows have no
independent mode declaration. An empty declaration must not be interpreted as
zero supported upstream features. Modes not listed as supported remain unverified;
the explicit rejection column is not an exhaustive list of every possible mode.

| Provider | Declared normalized modes | Explicitly unsupported modes |
| --- | --- | --- |
| openai | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS`, `RESPONSES_TOOLS` | None explicitly listed |
| anyscale | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | `RESPONSES_TOOLS` |
| together | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | `RESPONSES_TOOLS` |
| databricks | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | `RESPONSES_TOOLS` |
| deepseek | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | `RESPONSES_TOOLS` |
| openrouter | `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | `RESPONSES_TOOLS` |
| anthropic | `TOOLS`, `JSON`, `JSON_SCHEMA`, `PARALLEL_TOOLS` | `MD_JSON` |
| genai | `TOOLS`, `JSON` | `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` |
| generative-ai | Alias to `genai` | None explicitly listed |
| gemini | `TOOLS`, `MD_JSON` | `JSON`, `JSON_SCHEMA`, `PARALLEL_TOOLS`, `RESPONSES_TOOLS` |
| cohere | `TOOLS`, `JSON_SCHEMA`, `MD_JSON` | `PARALLEL_TOOLS`, `RESPONSES_TOOLS` |
| perplexity | `MD_JSON` | `JSON`, `TOOLS`, `JSON_SCHEMA`, `PARALLEL_TOOLS`, `RESPONSES_TOOLS` |
| xai | `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | `RESPONSES_TOOLS` |
| groq | `TOOLS`, `JSON_SCHEMA`, `MD_JSON` | `PARALLEL_TOOLS`, `RESPONSES_TOOLS` |
| mistral | `TOOLS`, `JSON_SCHEMA`, `MD_JSON` | `PARALLEL_TOOLS`, `RESPONSES_TOOLS` |
| fireworks | `TOOLS`, `JSON_SCHEMA`, `MD_JSON` | `PARALLEL_TOOLS`, `RESPONSES_TOOLS` |
| cerebras | `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` | `RESPONSES_TOOLS` |
| writer | `TOOLS`, `JSON_SCHEMA`, `MD_JSON` | `PARALLEL_TOOLS`, `RESPONSES_TOOLS` |
| bedrock | `TOOLS`, `TOOLS_STRICT`, `JSON_SCHEMA`, `MD_JSON` | `PARALLEL_TOOLS`, `RESPONSES_TOOLS` |
| vertexai | `TOOLS`, `MD_JSON`, `PARALLEL_TOOLS` | `JSON`, `JSON_SCHEMA`, `RESPONSES_TOOLS` |
| azure_openai | Alias to `openai` | None explicitly listed |
| ollama | Alias to `openai` | None explicitly listed |
| litellm | Delegated adapter; no modes declared here | None explicitly listed |

The `Provider.GOOGLE` and `UNKNOWN` enum values are not independent specs.
[Automatic routing](../../instructor/auto_client.py) and public compatibility
constructors are additional surfaces; this inventory does not promise that all
constructor/API/mode combinations are interchangeable. Registry tests in
[test_provider_specs.py](../../tests/v2/test_provider_specs.py) and
[test_provider_modes.py](../../tests/v2/test_provider_modes.py) cover declarations,
normalization and selected clients. They are not a live cross-product run.

## Provider/API contract matrix

This matrix explicitly separates **local wire contracts** from the shared live
suite. `L` means real SDK plus loopback HTTP in this PR; `C` means the existing
shared live suite contains a selected-mode asynchronous test; `U` means no
independent real-SDK wire proof was established in this audit. `U` does not mean
the feature is unsupported or that no unit tests exist. `C` is source inventory,
not evidence of execution. Every `C` cell is conditional on credentials, SDK
availability, model configuration and capability skips.

| Provider / API | Sync | Async | Streaming | Retry | Raw response | HTTP error |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI / Chat Completions, TOOLS | U | C | C | C* | C† | U |
| OpenAI / Responses, RESPONSES_TOOLS | L | L | U | U | L | L (400 cause) |
| Anthropic / Messages, ANTHROPIC_TOOLS | U | C | C | C* | C† | U |
| Google GenAI / generate_content, GENAI_STRUCTURED_OUTPUTS | U | C | C | C* | C† | U |
| Cohere / chat, COHERE_TOOLS | U | C | C | C* | C† | U |
| xAI / native chat, XAI_TOOLS | U | C | C | C* | C† | U |
| Mistral / chat, MISTRAL_TOOLS | U | C | C | C* | C† (completion only) | U |
| Cerebras / chat, CEREBRAS_TOOLS | U | C | C | C* | C† (completion only) | U |
| Fireworks / chat, FIREWORKS_TOOLS | U | C | C | C* | C† (completion only) | U |
| Writer / chat, WRITER_TOOLS | U | C | C | C* | C† (completion only) | U |
| Perplexity / OpenAI chat, PERPLEXITY_JSON | U | C | Capability-skipped | C* | C† (completion only) | U |
| Groq / chat | U | U | U | U | U | U |
| Bedrock / Converse | U | U | U | U | U | U |
| Legacy Gemini / generate_content | U | U | U | U | U | U |
| Vertex AI / generate_content | U | U | U | U | U | U |
| Anyscale, Together, Databricks, DeepSeek, OpenRouter / compatible chat | U | U | U | U | U | U |
| Azure OpenAI, Ollama / delegated OpenAI; LiteLLM / delegated completion | U | U | U | U | U | U |

`C*`: [test_retries.py](../../tests/llm/test_core_providers/test_retries.py)
passes `max_retries` and checks successful output. It does **not** force a failed
attempt or assert that a reask occurred. `C†`:
[test_response_modes.py](../../tests/llm/test_core_providers/test_response_modes.py)
checks a non-null completion and, where capability policy allows,
`response_model=None`; it does not assert the exact SDK class or wire payload.
The [shared configuration](../../tests/llm/shared_config.py) lists one mode/model
per provider. [Capability policy](../../tests/llm/test_core_providers/capabilities.py)
controls skips, including union streaming. It is test policy, not an authoritative
upstream feature specification. These cells do not cover other modes in the first
table.

The new [Responses contracts](../../tests/v2/test_responses_sdk_contract.py) verify
sync/async `/v1/responses` routing, `messages` to `input`, token-limit field
translation, function-tool schema shape, parsing a typed function-call response,
raw SDK response types, and preservation of a real SDK `BadRequestError` as the
retry exception cause. Each test asserts exactly one local request. SDK retries
are disabled; this does not certify validation retries, SSE, rate limits, refusals,
incomplete output, or upstream schema acceptance. The protocol references are
[OpenAI Responses create](https://developers.openai.com/api/reference/typescript/resources/beta/subresources/responses/methods/create)
and the [official Python SDK](https://github.com/openai/openai-python#handling-errors).

## Other existing evidence and untested areas

There is substantial unit/handler coverage outside the deliberately narrow `L`
cells. The following paths are inspection entry points, not claims that every test
uses a real SDK or that these suites were all executed for this PR:

| Provider/API | Offline test entry points | Additional live selection / gap |
| --- | --- | --- |
| OpenAI chat/Responses | [handlers](../../tests/coverage/test_openai_handlers_coverage.py), [streaming](../../tests/v2/test_openai_streaming.py), [Responses routing](../../tests/v2/test_openai_responses_client.py) | `tests/llm/test_openai`; Responses routing previously used mocks |
| Anthropic | [handlers](../../tests/coverage/test_anthropic_handlers_coverage.py), [support](../../tests/coverage/test_anthropic_support_coverage.py) | `tests/llm/test_anthropic`: multimodal, reasoning, system |
| GenAI | [integration](../../tests/v2/test_genai_integration.py), [coverage](../../tests/coverage/test_genai_coverage.py) | `tests/llm/test_genai`; offline integration includes substituted clients |
| Legacy Gemini | [coverage](../../tests/coverage/test_gemini_coverage.py) | `tests/llm/test_gemini`; legacy SDK availability is distinct from GenAI |
| Cohere | [coverage](../../tests/coverage/test_cohere_coverage.py) | Shared core suite only in Test workflow |
| xAI | [client](../../tests/coverage/test_xai_client_coverage.py), [handlers](../../tests/coverage/test_xai_handlers_coverage.py) | Shared core suite; native SDK requires Python 3.10+ |
| Mistral | [coverage](../../tests/coverage/test_mistral_coverage.py) | Shared core suite |
| Writer | [clients](../../tests/coverage/test_provider_clients_coverage.py), [handlers](../../tests/v2/test_writer_handlers.py) | `tests/llm/test_writer` |
| Vertex AI | [coverage](../../tests/coverage/test_vertexai_coverage.py), [runtime](../../tests/v2/test_vertexai_runtime.py) | `tests/llm/test_vertexai`; Google API key alone does not prove cloud authentication |
| Bedrock | [coverage](../../tests/coverage/test_bedrock_coverage.py), [conversion](../../tests/llm/test_bedrock) | No dedicated live-provider matrix job in Test workflow |
| Groq, Cerebras, Fireworks, compatible/delegated providers | [clients](../../tests/coverage/test_provider_clients_coverage.py), [compat handlers](../../tests/v2/test_openai_compat_handlers.py), [routing](../../tests/v2/test_auto_client_deterministic.py) | Only some receive shared-core or documented-model checks; compatible payloads do not certify remote services |

[Documented-model checks](../../tests/llm/test_current_doc_models.py) have their own
credential skips and are reported separately. `tests/v2/test_provider_modes.py`
also contains optional integration paths; it is not the dedicated live-provider
matrix selection. Statement/branch coverage from `tests/coverage` combines SDK
objects, test doubles and substituted functions. It cannot replace the wire or
live contract matrix above. Packaging and minimum/maximum SDK compatibility are
owned by a separate workflow/task; no dependency range claim is added here.

## Reading CI evidence

The [Test workflow](../../.github/workflows/test.yml) appends provider evidence to
GitHub's job summary even after test failures. Configuration-gated skips remain
successful but visibly **untested**. Existing exit-code-5 handling remains
non-failing with a warning; no tests collected is not validation. Missing reports
are **unknown**, never invented zero counts. A real empty report has zero outcomes;
an all-skipped report has skip counts and no passing evidence. Skipped upstream
jobs may prevent dependent jobs from starting; those jobs cannot generate a summary.

Initial, retry and documented-model reports are separate. A successful rerun never
erases the initial failure from the summary. Counts are JUnit outcomes, not unique
tests across attempts or API call counts. A failed teardown may be a separate
JUnit error. Expected failures count as skipped; unexpected non-strict passes
follow pytest's JUnit pass semantics. Skip categories are coarse message-based
labels, with uncategorized reasons in `pytest -rs` logs. Failure bodies and secret
values are not copied into summaries. Tests omitted during dynamic collection or
selection are not inferred from XML; the configuration presence list and source
matrix must be read alongside the outcome counts.

Offline/full-coverage job success is not live evidence: marker/name selection is
not a network sandbox, some legacy live tests lack `llm` markers, and some tests
fetch public resources. Do not add provider credentials to those runs to measure
coverage. This PR changes reporting and adds local contracts, not the paid-call
selection, retries, coverage thresholds, or required gates.
