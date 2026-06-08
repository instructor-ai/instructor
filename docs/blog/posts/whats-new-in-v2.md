---
authors:
  - jxnl
categories:
  - instructor
comments: true
date: 2026-05-11
description: "A technical tour of Instructor v2: provider-owned dispatch, executable capability contracts, compatibility facades, streaming, and typed results."
draft: false
slug: whats-new-in-instructor-v2
tags:
  - Instructor
  - Structured Outputs
  - Providers
  - Python
---

# What's new in Instructor v2?

Instructor's public job is simple: pass a Pydantic model to an LLM client and
get validated Python data back. Instructor v2 keeps that API, but changes how
the library implements it. Provider packages, or an explicitly shared
wire-compatible implementation, now own the SDK construction, wire format,
streaming events, tool schema, and validation reask payload.

This is not only a directory move. It changes the unit of correctness. In v2,
support for a feature is declared for a provider, registered by that
provider's handlers, and exercised by tests generated from the same
declaration.

<!-- more -->

## The user-facing path stays short

The common call still looks like this:

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4o-mini")
user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: Jason is 36"}],
)
```

Async construction remains the same decision expressed once:

```python
async_client = instructor.from_provider(
    "anthropic/claude-sonnet-4-6",
    async_client=True,
)
user = await async_client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: Jason is 36"}],
)
```

The technical path behind those calls is now:

```text
"anthropic/claude-sonnet-4-6"
  -> alias lookup in ProviderSpec
  -> lazy import of instructor.v2.providers.anthropic.client
  -> build_from_model(...)
  -> handler lookup for Provider.ANTHROPIC + Mode
  -> request preparation, response parsing, streaming, and reasks
  -> typed User result
```

The first three steps construct a client. The provider and mode handler owns
everything that changes the provider's payload or response interpretation.

## Why a provider-owned architecture?

Structured output is not one transport feature. It includes:

- converting a Pydantic schema into the provider's tool or JSON-schema shape
- translating `Image`, `Audio`, and `PDF` values into SDK payloads
- extracting JSON fragments from each provider's streaming events
- deciding which validation error message or tool result to send on a retry
- initializing sync and async SDK clients with provider-specific credentials

Those operations may begin with the same Python model, but they do not end in
the same request. Anthropic tools use `input_schema`. OpenAI tools use a
function object. Google GenAI builds `types.FunctionDeclaration` and
`GenerateContentConfig`. Vertex AI and legacy Gemini have different SDK
surfaces again.

V2 puts those decisions under:

```text
instructor/v2/providers/<provider>/
```

Shared core code still owns shared mechanics: the client wrapper, retry
orchestration, validation primitives, the handler registry, common media
value types, and compatibility entrypoints. It no longer needs to be the
place where every provider's wire-format branch accumulates.

## `from_provider()` is now dispatch, not construction policy

The model-string factory previously had to know how to build many kinds of
SDK clients directly. In v2, `auto_client.py` performs four operations:

1. Split `"provider/model"` into an alias and model name.
2. Resolve the alias through `ProviderSpec`.
3. Lazily import the builder module named by that spec.
4. Call its `build_from_model(...)` function with common arguments.

A simplified form of the implementation is:

```python
provider_enum = ALIAS_TO_PROVIDER.get(provider_name)
spec = PROVIDER_SPECS.get(provider_enum)
module = importlib.import_module(spec.model_builder_module)

return module.build_from_model(
    provider=spec.provider,
    model_name=model_name,
    async_client=async_client,
    mode=mode,
    api_key=api_key,
    kwargs=kwargs,
)
```

That reduction is measurable in code size: the v2 `auto_client.py` is now 181
lines. More importantly, adding or fixing provider construction no longer
means extending a central factory branch.

V2 does share construction when the behavior is genuinely identical.
OpenAI-compatible endpoints can use `compatible_model_builder(...)`, a
higher-order builder that handles an OpenAI wire-compatible client, base URL,
API key environment variable, and default mode. Providers that need special
credentials or client classes, such as Azure OpenAI or Databricks, retain
specialized construction inside the OpenAI provider builder instead of being
forced into the generic path.

## `ProviderSpec` is executable documentation

Provider support is represented by an immutable specification. The important
parts are:

```python
@dataclass(frozen=True)
class ProviderCapabilities:
    partial_stream_modes: tuple[Mode, ...] = ()
    iterable_stream_modes: tuple[Mode, ...] = ()
    multimodal_inputs: tuple[str, ...] = ()
    explicit_parallel_tools: bool = False


@dataclass(frozen=True)
class ProviderSpec:
    aliases: tuple[str, ...]
    handler_module: str | None
    supported_modes: tuple[Mode, ...]
    unsupported_modes: tuple[Mode, ...]
    legacy_modes: Mapping[Mode, Mode]
    client_module: str | None
    sdk_module: str | None
    capabilities: ProviderCapabilities
    builder_module: str | None = None
```

This is a contract, not a marketing list. It distinguishes four different
questions:

- Can the provider construct an Instructor client?
- Which core modes can its handlers execute?
- Which older mode names should normalize with compatibility behavior?
- Which user-visible features have conformance coverage?

For example, the current contract declares:

| Provider path | Modes in the contract | Streaming contract | Typed multimodal inputs |
| --- | --- | --- | --- |
| `openai/...` | tools, JSON, JSON schema, Markdown JSON, parallel tools, Responses tools | partial and iterable, including Responses tools where declared | image, audio, PDF |
| `anthropic/...` | tools, JSON, JSON schema, parallel tools | partial for tools/JSON/JSON schema; iterable for tools | image, PDF |
| `google/...` | tools, JSON | partial and iterable for both declared modes | image, audio, PDF |
| `gemini/...` | tools, Markdown JSON | partial only | none declared |
| `vertexai/...` | tools, Markdown JSON, parallel tools | partial for tools and Markdown JSON | none declared |
| `xai/...` | tools, JSON schema, Markdown JSON, parallel tools | partial and iterable only for the declared streaming modes | none declared |

An empty capability is meaningful. It means the shared v2 conformance suite
does not promise that feature for that provider.

## Handlers own request, response, stream, and reask behavior

Modes are registered against a provider:

```python
@register_mode_handler(Provider.ANTHROPIC, Mode.TOOLS)
class AnthropicToolsHandler(AnthropicHandlerBase):
    ...
```

A mode handler is the local place for four related operations:

- `prepare_request(...)` turns a response model and user kwargs into the SDK
  request shape.
- `parse_response(...)` converts the provider response into the requested
  Pydantic type.
- streaming extractors expose provider event chunks to `Partial` and
  `Iterable` processing.
- `handle_reask(...)` constructs the provider's correction request after
  validation fails.

The retry loop remains shared because retry orchestration is generic. The
reask payload is local because sending an Anthropic tool error is not the same
operation as sending a GenAI function response or an OpenAI tool message.

The ownership boundary is visible in specific implementations:

| Provider family | Provider-owned work in v2 |
| --- | --- |
| Anthropic | Anthropic tool schema generation, parallel tool schemas, media conversion callbacks, streaming extraction, and reask payloads |
| OpenAI and compatible handlers | OpenAI function/JSON schema shapes, Responses tool events, OpenAI media conversion, and compatible mode registration |
| Google GenAI | `GenerateContentConfig`, function declarations, content/media conversion, stream extraction, and GenAI reasks |
| Mistral | Mistral message conversion and stream parsing, with explicit compatible media encoder reuse where the wire shape matches |

There is still intentional reuse inside provider families. Several
OpenAI-compatible providers register the OpenAI handler implementation rather
than copying equivalent wire logic. The GenAI handler uses Google
schema/message utilities currently located in
`instructor.v2.providers.gemini.utils`. That is Google-specific reuse, not a
claim that the two SDKs expose identical public behavior: `google/...`,
`gemini/...`, and `vertexai/...` have separate provider specs and separate
declared capabilities. The `google/...` path targets `google.genai`;
`gemini/...` remains the legacy `google.generativeai` integration, and
`generative-ai/...` is a compatibility alias for the GenAI builder.

## Multimodal values stay common; encoders do not

Users should not need a new `Image` class for every provider. V2 therefore
keeps common media value types in core. The conversion boundary changed:
active handlers pass provider-local encoders into the shared message walker.

For example, Anthropic passes `media_to_anthropic` and `image_from_params`
when converting messages. OpenAI passes its own encoders. GenAI produces
Google `Part` content through its provider module. This preserves a common
input API without asking core code to produce every provider's final wire
payload.

Compatibility wrappers remain in shared modules for public imports that
already exist. They forward into provider-owned implementations; the active
Anthropic, Gemini, GenAI, and OpenAI-compatible request paths updated in this
refactor no longer need a core compatibility schema surface to build provider
requests.

## Compatibility is lazy, not eager

V2 is intended to be a practical upgrade, so older provider factory and
utility imports continue to exist. The facade modules use module-level
`__getattr__` resolution:

```python
__getattr__ = make_getattr("anthropic", ("client",))
```

Importing `instructor.providers.anthropic.client` therefore does not import
the v2 Anthropic implementation or the optional Anthropic SDK until a
forwarded symbol is actually accessed. Subprocess tests block optional SDK
imports and verify that legacy client facades can still be imported.

This boundary also has a measurable import effect in fresh Python processes:

| Import | Before this refactor | After this refactor |
| --- | ---: | ---: |
| `import instructor` | `0.01s`, `17.5 MB` RSS | `0.01s`, `17.4 MB` RSS |
| `from instructor import Mode` | `0.03s`, `20.7 MB` RSS | `0.02s`, `19.5 MB` RSS |
| `import instructor.providers.anthropic.client` | `0.55s`, `79.2 MB` RSS | `0.02s`, `19.6 MB` RSS |

These numbers are an implementation comparison, not a promise that every
machine will have identical timings. The guarantee enforced in tests is the
architectural one: compatibility imports do not eagerly require optional
provider SDKs.

## Tests now follow the feature contract

The test suite reads the same provider specifications used by runtime
dispatch. A shared matrix derives:

- provider and mode registration cases
- partial streaming cases
- iterable streaming cases
- typed multimodal cases
- explicit parallel-tools cases

The conformance tests then execute each advertised feature. If a provider
adds a streaming mode to `ProviderCapabilities`, it joins the streaming
contract tests. If it removes that declaration, the public feature claim is
removed with it. Unique SDK request shapes can still have provider-specific
tests; repeated capability bookkeeping no longer needs a hand-maintained list
per test file.

These are deterministic contract tests: they use representative stream events
and media values to prove handler registration and conversion behavior without
calling hosted models. They do not by themselves claim that every provider
and model performs equally well in production.

The migration also adds two guardrails beyond runtime tests:

- public sync and async factory inference is checked with `assert_type(...)`
  and Pyright, including `from_provider(..., async_client=True)`;
- deterministic lazy-import tests protect the optional-SDK boundary described
  above.

## Benchmarking model and mode combinations

V2 also includes a live benchmark example for the question deterministic
contracts cannot answer: for a particular extraction task, which configured
model and mode combination is fast and reliable?

```bash
uv run python examples/v2-model-mode-benchmark/run.py --trials 3
```

The runner derives mode cells from `ProviderSpec`, uses each provider's
configured default model when one exists, and emits one result per model/mode
cell. Cells are reported as skipped when the model has no configured default,
its optional SDK is unavailable, or its required API-key environment variable
is absent. Providers that authenticate through a cloud credential chain, such
as Bedrock or Vertex AI, require explicit `--allow-cloud-auth` so the runner
does not guess whether ambient credentials are valid. That makes the default
run safe in a partial local environment without turning missing keys into
failed comparisons.

To compare selected models rather than the defaults:

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

The benchmark records success rate and latency for a small typed extraction
probe, then ranks completed cells by correctness first and median latency
second. It is a harness for comparing a supplied workload, not a universal
model leaderboard; a meaningful decision should replace or extend the probe
with representative application examples.

## What upgrading code requires

For most applications, no architectural knowledge is required. Existing
factory paths remain compatibility entrypoints, while new code can prefer the
model-string form:

```python
# Existing factory style remains supported.
client = instructor.from_anthropic(anthropic_client)

# Recommended v2 construction path.
client = instructor.from_provider("anthropic/claude-sonnet-4-6")
```

Older provider-specific modes normalize to the smaller v2 mode vocabulary
where a mapping exists. New code should select core modes such as
`Mode.TOOLS`, `Mode.JSON_SCHEMA`, `Mode.MD_JSON`, or
`Mode.PARALLEL_TOOLS`, and check the integration guide for provider-specific
support before depending on streaming or multimodal behavior.

For contributors, the rule is stricter:

1. Put provider SDK construction in that provider's client module.
2. Put wire schemas, streaming events, and reask payloads in that provider's
   handler or helper modules.
3. Declare supported user-visible behavior in `ProviderSpec`.
4. Add shared conformance coverage through the manifest, and add one-off tests
   only for genuinely unique behavior.
5. Keep core exports only when they represent shared machinery or compatibility
   that users already import.

That is the technical change in v2: Instructor still presents one typed
structured-output workflow, while its implementation stops pretending every
provider reaches that workflow through the same protocol.
