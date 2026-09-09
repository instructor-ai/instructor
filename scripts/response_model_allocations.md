# Response-model allocation evidence

This probe investigates [issue #2603](https://github.com/567-labs/instructor/issues/2603)
at baseline `6969bb6849c49bf742fcdbfec7e969cfd5046274`. No runtime caching change is
included: the measurements demonstrate repeated construction and bounded schema
cache retention, not a sustained leak of generated classes in these paths.

## Reproduce offline

From the repository root in an environment with Instructor dependencies installed:

```sh
PYTHONPATH=. python scripts/benchmark_response_model_allocations.py > allocations.json
PYTHONPATH=. python scripts/benchmark_response_model_allocations.py --scenario stable --retain-results > held.json
PYTHONPATH=. python scripts/benchmark_response_model_allocations.py --scenario dynamic --batch-size 1000 > dynamic.json
```

Each scenario warms ten requests, clears the OpenAI schema cache, and collects
garbage before tracing. Three batches of 300 calls run by default. Preparation
alone and preparation plus OpenAI schema generation are measured separately.
Only weak references to generated results are kept. The optional `--retain-results`
control reproduces the issue's `held.append(prepared)` behavior. Dynamic inputs
have distinct class names and cycle through one to eight integer fields; their
construction cost is included. Nested input contains a list of two-field models.

JSON records traced bytes before/after GC, peak traced bytes,
live generated classes, cache hits/misses/capacity, and retention after releasing
held results and clearing the cache. This probe does not measure timing; the
separate generic benchmark owns timing and performance comparisons. This probe's
JSON output and Python/Pydantic labels remain standalone because its cache-clear,
weak-reference, and retained-results controls are specific to this investigation.
No provider requests are made.

## Observations

The byte measurements below were recorded with probe revision `e89c45a7` before
removing its unused elapsed-time field. Across all nine scenarios listed below,
both environments retained 0 / 0 / 0 generated classes after GC at 300 / 600 / 900
preparation calls, or 256 / 256 / 256 with schema generation. Clearing the schema
cache released all generated classes in every scenario.

Post-GC Python allocations remain nonzero even when classes are collectible.
For preparation plus schema generation, rounded KiB above the warmed baseline:

| Scenario | Python 3.11.11, Pydantic 2.11.7: 300 / 600 / 900 | Python 3.13.11, Pydantic 2.12.5: 300 / 600 / 900 |
| --- | --- | --- |
| Stable (2 fields) | 2646 / 2698 / 2751 | 2668 / 2696 / 2732 |
| Wide (8 fields) | 4162 / 4252 / 4322 | 4894 / 4921 / 4958 |
| Nested | 3231 / 3259 / 3296 | 3267 / 3295 / 3332 |
| list[Model] | 3461 / 3512 / 3580 | 3168 / 3197 / 3235 |
| Iterable[Model] | 3467 / 3527 / 3580 | 3168 / 3196 / 3234 |
| TypedDict | 11901 / 11948 / 11998 | 5027 / 5055 / 5091 |
| list[TypedDict] | 6037 / 6105 / 6192 | 5478 / 5515 / 5570 |
| int | 2545 / 2571 / 2623 | 2255 / 2283 / 2320 |
| Dynamic | 13348 / 13399 / 13488 | 6792 / 6821 / 6857 |

Before correcting the measurement, the retained-results control on Python 3.11
holds 300 / 600 / 900 classes and 3014 / 5732 / 8369 KiB with schema generation.
After removing that artificial retention, the stable row above retains only 256
classes. Releasing the control's held list and clearing the schema cache also
releases every generated class. This comparison changes the measurement, not
Instructor's runtime. Repeated construction and zero schema-cache hits for fresh
wrappers are real costs; they do not establish unbounded class retention.

Longer Python 3.13/Pydantic 2.12 runs at 1,000 / 2,000 / 3,000 calls retained
256 / 256 / 256 generated classes with schema generation and zero after clearing
the cache. Stable-model post-GC allocations were 2732 / 2733 / 2733 KiB; dynamic
models were 6856 / 6856 / 6856 KiB. Preparation alone retained zero generated
classes after GC in both longer runs.

The first environment uses the repository's locked Pydantic version. The second
uses an isolated installation with Pydantic 2.12.5; other dependencies were
resolved independently, so this is not a controlled Python-only comparison.
Tracemalloc excludes some native allocations and does not measure RSS. Byte
drift includes probe bookkeeping, allocator/library caches, and schema diversity;
the data do not prove a flat heap or rule out other long-running memory problems.

## Identity, invalidation, and competing proposals

- `response_schema`/`openai_schema` currently create a fresh subclass. Public
  `instructor.openai_schema`, `instructor.function_calls`, and
  `instructor.processing.function_calls` resolve to the same implementation.
  `instructor.utils.core.prepare_response_model` is also a compatibility alias.
- Fresh preparation observes source field changes followed by `model_rebuild`;
  mutations to an earlier wrapper do not become mutations to a later wrapper's
  field metadata. Already-prepared models retain their existing identity.
- `generate_openai_schema` has an existing 256-entry strong-reference LRU.
  Eviction or `generate_openai_schema.cache_clear()` releases its references.
  When explicitly reusing and rebuilding a prepared class, callers must clear
  this schema cache to refresh its cached schema. Automatic invalidation is not
  added here. A bounded entry count is not a fixed byte ceiling.
- [PR #2605](https://github.com/567-labs/instructor/pull/2605) stores a strong
  subclass value in a `WeakKeyDictionary` keyed by its base class. The value's
  MRO and `__wrapped__` retain the key. A direct local probe leaves the input
  class alive after dropping the caller reference and collecting garbage;
  clearing that dictionary releases it. Its cleanup test has no collection
  assertion. This proposal can add unbounded retention for dynamic models.
  Reviewed head: `7e7d33a821b1b1593257a820ab52e656317405a8`.
- [PR #2612](https://github.com/567-labs/instructor/pull/2612) bounds its model
  cache at 1024 entries, but hits skip current async-validator rejection and
  reuse stale wrappers after source mutations/rebuilds. Catching `TypeError`
  around the entire cached call also retries errors raised inside preparation,
  not only unhashable cache keys.
  Reviewed head: `872dc70732bfb39cbce34201d73ec4f0394d57f5`.

The post-PR review loaded both exact candidate modules locally alongside the
baseline runtime (Python 3.11.11/Pydantic 2.11.7). PR #2605 kept its dynamic input
alive after GC and released it only after dictionary clearing. PR #2612 returned
default `1` after rebuilding the source with default `2`, accepted a newly added
nested async validator that baseline preparation rejected, and invoked a real
Pydantic schema hook that raises `TypeError` twice versus baseline's once.

Memoization requires a separate contract for mutable model metadata, nested
rebuilds, per-request streaming flags, and current async-validator rejection.
The new tests exercise these observable preparation contracts and verify
collection with weak references, including eviction of dynamic input classes.
They intentionally avoid byte-count thresholds and do not assert an RSS bound.
