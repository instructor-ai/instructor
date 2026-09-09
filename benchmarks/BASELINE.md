# Initial offline baseline

Measured checkout: `1d8fca7851d7a9397d0618acf9a9f684e5d5223a` (dirty: `False`).
Harness SHA-256: `6702fd0087ab1f9961209898cbf4fb01d4472f0fef494d5e70ba2da498b34b69`.
Started: 2026-09-09T03:13:20.012899+00:00.

Environment: Python 3.11.11, macOS-15.3-arm64-arm-64bit, 10 logical CPUs.
Installed distribution metadata: instructor 1.16.1, pydantic 2.11.7, pydantic-core 2.33.2, openai 2.6.0, jiter 0.10.0.

Instructor itself was imported from the measured checkout, not the installed
1.16.1 distribution. This used an existing environment, not a new locked
installation. Other work may have been active on the shared host; it was not reserved for benchmarking.
Treat this as an initial descriptive baseline, not an optimization comparison.

Command (run with the interpreter from that environment):

```sh
python scripts/benchmark_local.py --samples 7 --iterations 5 \
  --memory-batches 4 --memory-iterations 75 --memory-stream-iterations 5 \
  --output benchmarks/baseline.json
```

Three warmup operations; 128-character chunks; seven timing samples of five
operations each; four GC-aware memory batches of 75 preparation/schema calls
or five full streams (300 schema calls; 20 streams per mode). First-yield
statistics use seven single-stream observations. Import uses seven fresh processes.

[Raw measurements](baseline.json) include every sample and memory checkpoint.
[Method and regression policy](README.md) define exclusions and comparison rules.

## Local timing

Cold import wall median: 1.860 ms (range 1.750–2.205 ms).

All table values are median wall microseconds per operation, except first yield
(microseconds from stream entry to first model). They include local Pydantic/SDK
work but no provider/network latency. Async totals include event-loop driving.

| Operation | Small | Medium | Large |
| --- | ---: | ---: | ---: |
| `prepare` | 194.417 | 197.883 | 197.200 |
| `schema_pydantic` | 149.525 | 238.042 | 567.217 |
| `schema_cached` | 0.683 | 0.925 | 0.958 |
| `prepare_schema` | 362.367 | 450.000 | 871.400 |
| `parse_pydantic` | 4.150 | 15.425 | 292.400 |
| `parse_instructor` | 51.583 | 63.892 | 368.292 |
| `stream_sync` | 12.633 | 560.700 | 332,444.958 |
| `stream_async` | 57.392 | 591.108 | 341,167.892 |
| `stream_sync` first yield | 4.041 | 14.958 | 37.291 |
| `stream_async` first yield | 5.417 | 17.167 | 39.125 |

## Sustained traced allocations

Post-GC deltas in KiB, relative to tracing start after warmup. Each cell lists
four checkpoints: 75 / 150 / 225 / 300 preparation/schema calls, or
5 / 10 / 15 / 20 full streams. Returned objects are discarded.
These include Python cache retention and small harness bookkeeping, and exclude
native allocations and allocations made before tracing. They are not RSS.
Streaming uses fewer repetitions because a 300-stream allocation worker took
over five minutes on this host; no inputs, GC checkpoints or timing gates were
changed. Longer retention tests should increase the explicit stream count.

| Case | Payload bytes / chunks | Prepare + schema | Sync stream | Async stream |
| --- | ---: | --- | --- | --- |
| small | 39 / 1 | 865.0 / 1,675.8 / 2,488.6 / 2,840.7 | 0.2 / 0.4 / 0.7 / 1.0 | 0.4 / 0.7 / 1.0 / 1.2 |
| medium | 1,727 / 14 | 977.4 / 1,905.9 / 2,833.1 / 3,229.2 | 0.2 / 0.4 / 0.7 / 1.0 | 0.4 / 0.7 / 1.0 / 1.2 |
| large | 63,415 / 496 | 1,459.6 / 2,861.0 / 4,271.4 / 4,855.2 | 0.2 / 0.4 / 0.7 / 1.0 | 0.4 / 0.7 / 1.0 / 1.2 |

The preparation series passes the current 256-entry provider schema cache bound
only at its last checkpoint. This short series cannot demonstrate a plateau or
an unbounded leak. The dedicated allocation investigation owns weak-reference
class counts, cache clearing, dynamic models and retained-result controls.

## Validation and scope

- Seven harness tests pass, including all three input sizes, equal sync/async
  final values, cyclic return-value collection, worker errors and tracing cleanup.
- Related offline selection: 76 passed, two provider-backed summary extraction
  tests deselected. An initial broader run hit DNS failures in those two tests;
  no provider response was received. They are not counted as offline passes.
- Focused Ruff lint, Ruff format and explicit-file ty checks passed. The type
  check used a temporary config with no source exclusions because this repository
  normally excludes scripts and tests from ty.
- No runtime modules changed. Before this PR, the existing import script took
  one traced sample and the streaming example included provider latency. After
  this PR, offline CPU/wall, first-yield and post-GC measurements are available.
  There is no runtime before/after speedup claim.
- This covers one Python/dependency environment and synthetic JSON-mode inputs.
  It does not establish cross-version/provider performance, real transport
  latency, cancellation behavior or fairness under concurrent streams.

## Post-PR simplification review

The original baseline above remains tied to its recorded revision and checksum.
The later harness simplification removes repeated operation/metadata lists and
forwards worker flags from the recorded configuration; it does not re-label these
measurements as a new run. Sync and async timing loops remain explicit: the first
timestamp is captured after the parser yields a model, and the stream is drained
through final validation.

The separate streaming memory count remains at five streams per batch by default.
Schema preparation still runs enough times to exceed the current cache bound;
applying that repetition count to traced streaming made routine runs impractical.
No timing gates or allocation thresholds were added. The specialized
[GC probe in PR #2624](https://github.com/567-labs/instructor/pull/2624) retains its
cache/weak-reference/control diagnostics without a redundant timing layer.

Review validation: seven harness tests, focused Ruff lint/format and explicit-file
ty checks passed. A short medium/large probe completed with three samples, one
timed operation per sample, one warmup, and two memory batches of three schema
operations or one stream. All result summaries, source checksum and independent
memory operation counts were checked. This is a harness correctness check, not a
new performance comparison; the original baseline data remains unchanged.
