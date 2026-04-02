# Changelog

All notable changes to instructor are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [1.15.0] - Unreleased

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

See [GitHub releases](https://github.com/567-labs/instructor/releases/tag/v1.14.5) for details.
