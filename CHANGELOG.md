# Changelog

All notable changes to instructor are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

### Fixed
- **Templating (GenAI/VertexAI)**: `process_message` no longer crashes with `TypeError: Can't compile non template nodes` when multimodal messages contain image/URI/bytes Parts alongside `validation_context`. Non-text Parts (where `part.text` is `None`) now pass through unchanged. ([#2253](https://github.com/567-labs/instructor/issues/2253))
- **Retry**: `IncompleteOutputException` now propagates directly to the caller without being wrapped in `InstructorRetryException`, making `except IncompleteOutputException` catch blocks work as documented. Applies to both sync and async paths. ([#2273](https://github.com/567-labs/instructor/issues/2273))

---

## [1.15.1] - 2026-04-03

### Security
- **Bedrock**: Block remote HTTP(S) image URL fetching in `_openai_image_part_to_bedrock` — only `data:` URLs are now accepted, preventing SSRF via user-controlled image URLs
- **Bedrock/PDF**: Block remote URL and local file fetching in `PDF.to_bedrock` — only base64 data or `s3://` sources are now supported, preventing SSRF and local file disclosure

### Added
- **Hooks**: `completion:error` and `completion:last_attempt` handlers now receive `attempt_number`, `max_attempts`, and `is_last_attempt` as keyword arguments. Old-style handlers remain fully backward-compatible.
- **Anthropic**: `from_provider("anthropic/...")` now sets a `User-Agent: instructor/<version>` header on the Anthropic client

### Fixed
- **Anthropic usage**: Initialize usage correctly for `ANTHROPIC_REASONING_TOOLS` and `ANTHROPIC_PARALLEL_TOOLS` modes — previously fell through to OpenAI usage tracking with wrong field names
- **OpenRouter**: Use `reask_md_json` for `OPENROUTER_STRUCTURED_OUTPUTS` retries instead of `reask_default` (tool-call format), fixing malformed retry prompts
- **Templating**: Return `kwargs` unchanged instead of `None` in `handle_templating` when message list is empty or format is unrecognized; `process_message` also now returns the original message unchanged for unrecognized formats instead of `None`
- **`from_openai`**: Allow `Mode.JSON_SCHEMA` for the OpenAI provider — it was incorrectly blocked by the mode validation check
- **Bedrock**: Pass through `cachePoint` dicts in message content unchanged — previously raised `ValueError: Unsupported dict content for Bedrock`, breaking prompt caching (regression since v1.13.0)
- **Bedrock**: Allow `Mode.MD_JSON` in `from_bedrock`
- **Parallel tools**: `ParallelBase` generator now consumed into `ListResponse` in both sync and async paths, fixing `AttributeError` when setting `_raw_response` on a generator

---

## [1.15.0] - 2026-04-02

### Security
- Pin litellm to `<=1.82.6` to block compromised versions 1.82.7 and 1.82.8 ([#2219](https://github.com/567-labs/instructor/pull/2219))
- Make `diskcache` an optional dependency, removing it from all users' transitive dependency trees and mitigating CVE-2025-69872 ([#2211](https://github.com/567-labs/instructor/pull/2211))

### Fixed
- **Usage tracking**: Preserve `response.usage` subclass type (e.g. LiteLLM, Langfuse) when accumulating token counts across retries — fixes downstream `.get()` method loss ([#2217](https://github.com/567-labs/instructor/pull/2217), [#2199](https://github.com/567-labs/instructor/pull/2199))
- **Gemini**: Exclude `HARM_CATEGORY_IMAGE_*` safety categories from standard Gemini API calls — these are Vertex AI-only and caused `400 INVALID_ARGUMENT` errors ([#2174](https://github.com/567-labs/instructor/pull/2174))
- **Gemini**: Detect truncated responses (`finish_reason=MAX_TOKENS`) in `GENAI_STRUCTURED_OUTPUTS` mode and raise `IncompleteOutputException` immediately instead of retrying with malformed JSON ([#2232](https://github.com/567-labs/instructor/pull/2232))
- **`create_with_completion`**: Handle `List[Model]` response models that lack `_raw_response` attribute — previously raised `AttributeError`, now returns `None` for the completion ([#2167](https://github.com/567-labs/instructor/pull/2167))
- **Partial streaming**: Preserve default `Literal` field values (e.g. `type: Literal["Person"] = "Person"`) during streaming instead of emitting `None` before the field arrives ([#2204](https://github.com/567-labs/instructor/pull/2204))
- **Partial streaming**: Support PEP 604 union syntax (`str | int`) in `Partial` models on Python 3.10+ ([#2200](https://github.com/567-labs/instructor/pull/2200))
- **Validators**: Fix `allow_override=True` in `llm_validator` — the override branch was unreachable due to a misplaced assertion, so `fixed_value` was never returned ([#2215](https://github.com/567-labs/instructor/pull/2215))
- **Parallel tools**: `ParallelBase` responses now return `ListResponse` (consistent with `IterableBase`) instead of a raw generator with `_raw_response` set on it ([#2216](https://github.com/567-labs/instructor/pull/2216))
- **Multimodal**: Add missing `continue` in `convert_messages` after handling typed (`audio`/`image`) messages — previously fell through to `message["role"]` causing `KeyError` ([#2139](https://github.com/567-labs/instructor/pull/2139))
- **Anthropic**: Fix dead code path for `ANTHROPIC_REASONING_TOOLS` mode — the mode was shadowed by a duplicate `ANTHROPIC_TOOLS` check and never routed correctly ([#2140](https://github.com/567-labs/instructor/pull/2140))

### Added
- **Models**: Add Claude 4 (Opus, Sonnet, Haiku), OpenAI GPT-4.1 series, o3/o4 reasoning models, xAI Grok 3, and DeepSeek R1/V3 to `KnownModelName` type ([#2235](https://github.com/567-labs/instructor/pull/2235))

### Docs
- Update GitHub organization links in README from `instructor-ai` to `567-labs` ([#2149](https://github.com/567-labs/instructor/pull/2149))

### Tests / CI
- Fix `test_xai_optional_dependency` tests to use `monkeypatch` so they pass regardless of whether `xai-sdk` is installed
- Update deprecated Anthropic model names (`claude-3-5-haiku-latest` -> `claude-haiku-4-0-20250414`, `claude-3-7-sonnet-latest` -> `claude-sonnet-4-5-20250514`)
- Update deprecated OpenAI model names (`gpt-3.5-turbo` -> `gpt-4.1-mini`) across unit tests
- Update stale provider model strings in `shared_config.py`: Writer palmyra-x5, Fireworks llama-v3p3, Perplexity sonar-pro

---

## [1.14.5] - 2026-01-29

### Fixed
- **Google GenAI**: `thought_signature` is now preserved across validation retries for thinking models ([#2001](https://github.com/567-labs/instructor/pull/2001))
- **Metadata**: `pyproject.toml` author field corrected so PyPI correctly populates the `Author` field ([#2015](https://github.com/567-labs/instructor/pull/2015))
- **Deps**: Dev dependencies moved to the correct `[dependency-groups]` section in `pyproject.toml` ([#2030](https://github.com/567-labs/instructor/pull/2030))

---

## [1.14.4] - 2026-01-16

### Fixed
- **Responses API**: Validation errors during structured output parsing are now caught and retried correctly ([#2002](https://github.com/567-labs/instructor/pull/2002))
- **Google GenAI**: User-provided `GenerationConfig` labels and custom fields are no longer silently dropped when merging configs ([#2005](https://github.com/567-labs/instructor/pull/2005))
- **Google GenAI**: `SafetySettings` now applied correctly when request contains image content ([#2007](https://github.com/567-labs/instructor/pull/2007))
- **List responses**: Response wrappers no longer crash on attribute-style access ([#2011](https://github.com/567-labs/instructor/pull/2011))
- **`_raw_response`**: Attribute access on list response wrappers works correctly ([#2012](https://github.com/567-labs/instructor/pull/2012))

### Changed
- **`json_tracker`**: Sibling-heuristic algorithm simplified for improved partial-streaming reliability ([#2000](https://github.com/567-labs/instructor/pull/2000))

---

## [1.14.3] - 2026-01-13

### Added
- **Partial streaming**: Completeness-based streaming validation — fields are validated progressively rather than failing mid-stream ([#1999](https://github.com/567-labs/instructor/pull/1999))

### Fixed
- **Streaming reask**: `Stream` objects in reask handlers are now consumed correctly before retry, preventing stale-stream errors ([#1992](https://github.com/567-labs/instructor/pull/1992))

---

## [1.14.2] - 2026-01-13

### Fixed
- **Partial streaming**: Model validators now skip during partial streaming and run only once on the final complete object, preventing spurious errors ([#1994](https://github.com/567-labs/instructor/pull/1994))
- **Partial**: Infinite recursion with self-referential models (e.g. `TreeNode` with `children: List["TreeNode"]`) is now prevented ([#1997](https://github.com/567-labs/instructor/pull/1997))

### Tests / CI
- Provider tests skipped in CI when API secrets are not available ([#1990](https://github.com/567-labs/instructor/pull/1990))

---

## [1.14.1] - 2026-01-08

### Fixed
- **Google GenAI**: `cached_content` parameter now correctly forwarded to support Google context caching ([#1987](https://github.com/567-labs/instructor/pull/1987))

---

## [1.14.0] - 2026-01-04

### Added
- **Bedrock**: Document support — pass PDFs and text files directly to Bedrock models ([#1936](https://github.com/567-labs/instructor/pull/1936))

### Fixed
- **`from_provider()`**: Now respects the `base_url` keyword argument for OpenAI-compatible providers ([#1971](https://github.com/567-labs/instructor/pull/1971))
- **`from_provider()`**: Runtime `ImportError` exceptions are no longer masked, making misconfigured installs easier to diagnose ([#1975](https://github.com/567-labs/instructor/pull/1975))
- **Google GenAI**: `Union` types now allowed in structured output schemas ([#1973](https://github.com/567-labs/instructor/pull/1973))
- **Google GenAI**: `thinking_config` and additional user-provided `GenerationConfig` fields now correctly preserved ([#1972](https://github.com/567-labs/instructor/pull/1972), [#1974](https://github.com/567-labs/instructor/pull/1974))
- **Cohere**: Streaming and V2 API version detection issues resolved ([#1983](https://github.com/567-labs/instructor/pull/1983), [#1844](https://github.com/567-labs/instructor/pull/1844))
- **xAI**: Tools-mode validation fixed ([#1983](https://github.com/567-labs/instructor/pull/1983))
- **Exception handling**: Standardized across all providers ([#1897](https://github.com/567-labs/instructor/pull/1897))

### Changed
- **Type checker**: Switched from Pyright to `ty` for faster incremental type checking ([#1978](https://github.com/567-labs/instructor/pull/1978))
- **Provider factories**: `from_openai`, `from_anthropic`, etc. signatures standardized ([#1898](https://github.com/567-labs/instructor/pull/1898))

---

## [1.13.0] - 2025-11-03

### Added
- **Bedrock**: Image input support — converts OpenAI-style image parts to Bedrock's native format
- **`py.typed`**: Marker file restored for PEP 561 type-checking support ([#1868](https://github.com/567-labs/instructor/pull/1868))

### Fixed
- **`disable_pydantic_error_url()`**: Now correctly suppresses Pydantic validation error URLs via monkey-patching `ValidationError.__str__()` (environment variable approach had no effect post-import)
- **JSON mode**: JSON decode errors now trigger retry logic instead of surfacing as unhandled exceptions ([#1856](https://github.com/567-labs/instructor/pull/1856))
- **Gemini**: Streaming fixed for the Google GenAI SDK ([#1864](https://github.com/567-labs/instructor/pull/1864))
- **Gemini**: `HARM_CATEGORY_JAILBREAK` safety category and Anthropic `tool_result` content blocks now handled correctly ([#1867](https://github.com/567-labs/instructor/pull/1867))
- **Partial**: Fields with `default_factory` no longer retain the factory when made optional during streaming
- **OpenAI**: Dependency version constraint updated to support v2 ([#1858](https://github.com/567-labs/instructor/pull/1858))

---

## [1.12.0] - 2025-10-27

### Fixed
- **Python 3.13**: Compatibility issues and import path corrections in multimodal processing
- **Bedrock**: OpenAI-compatible models now correctly parse responses where reasoning appears before text content
- **Gemini**: `chunk.text ValueError` when `finish_reason=1` no longer crashes streaming
- **Gemini**: `thinking_config` no longer unintentionally passed to the tools helper
- **OpenAI**: `parse:error` hook now correctly fires for `InstructorValidationError`
- **JSON parsing**: Broken regex patterns removed from JSON extraction function
- **Cohere**: V2 API version detection improved ([#1844](https://github.com/567-labs/instructor/pull/1844))

---

## [1.11.3] - 2025-09-04

### Added
- **Hooks**: Hook combination via `__add__` / `combine()` — merge multiple hook handlers together
- **Hooks**: Per-call hooks — pass hooks directly to individual `.create()` calls without registering globally
- **Retry**: `InstructorRetryException` now tracks all failed attempts including exceptions and raw completions for better introspection
- **Docs**: `llms.txt` support via `mkdocs-llmstxt` plugin for AI/LLM consumers

### Fixed
- **`InstructorError.__str__()`**: Now correctly formats failed-attempt details
- **Retry**: Failed attempts propagated through reask handlers
- **Imports**: Backward compatibility imports restored for `function_calls` and `validators` modules

---

## [1.11.1] - 2025-08-27

### Changed
- Upgraded all dependencies to latest versions

---

## [1.11.0] - 2025-08-27

### Added
- **OpenRouter**: Provider support in `from_provider()` using `OPENROUTER_API_KEY`
- **LiteLLM**: Provider support in `from_provider()` ([#1723](https://github.com/567-labs/instructor/pull/1723))
- **xAI**: Provider utilities following standard provider structure ([#1728](https://github.com/567-labs/instructor/pull/1728))
- **Batch API**: In-memory batching support with improved error handling for OpenAI and Anthropic ([#1746](https://github.com/567-labs/instructor/pull/1746))
- **Hooks**: `completion:error` and `completion:last_attempt` hooks now fully implemented ([#1729](https://github.com/567-labs/instructor/pull/1729))

### Changed
- Codebase reorganized from flat structure to modular provider-based architecture ([#1730](https://github.com/567-labs/instructor/pull/1730))
- Provider-specific message conversion logic moved to dedicated handlers ([#1724](https://github.com/567-labs/instructor/pull/1724))

### Fixed
- Pydantic v2 deprecation warnings resolved by migrating from class `Config` to `ConfigDict` ([#1782](https://github.com/567-labs/instructor/pull/1782))

