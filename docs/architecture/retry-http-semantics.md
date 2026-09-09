# Retry execution and HTTP attempt measurements

Baseline: `6969bb6849c49bf742fcdbfec7e969cfd5046274` (Instructor 1.17.0).
Measured on macOS with Python 3.13.11, OpenAI 2.6.0, Anthropic 0.93.0,
HTTPX 0.28.1, and Tenacity 9.1.2 from the repository lockfile.

## Verified defect and correction

With a custom `Retrying(stop=stop_after_attempt(2))` policy and one SDK retry,
scripted responses `500, 500, 200` produce three HTTP requests and two Instructor
attempts. Before this patch, the first `completion:error` incorrectly emitted
`is_last_attempt=True`, even though the next Instructor attempt succeeded.
All four OpenAI/Anthropic sync/async regressions failed on this assertion before
the patch; the other 52 initial HTTP regressions passed.

The correction requires a known integer attempt limit before predicting finality
in the API-error hook. For a custom policy, `max_attempts=None` and
`is_last_attempt=False` until `completion:last_attempt` reports actual exhaustion.
This follows the existing conservative parse-error convention. It does not
inspect, evaluate twice, or alter a user's Tenacity policy. Exhaustion regression
cases also confirm the final hook remains true, with two Instructor attempts
and four HTTP requests. Integer-policy behavior is unchanged.

The runtime diff changes the same boolean grouping in the sync and async loops.
No retry counts, wait settings, defaults, exception types, provider usage
arithmetic, token-budget policy, or module ownership changed. This complements
[#2617](https://github.com/567-labs/instructor/pull/2617) and
[#2620](https://github.com/567-labs/instructor/pull/2620).

## Method and results

`tests/core/test_retry_http_semantics.py` uses the public Instructor clients,
real SDKs, and a threaded HTTP server bound to `127.0.0.1` on an ephemeral port.
It records each request body at the server, independently of Instructor hooks.
No mocks, provider credentials, or paid requests are used. SDK status retries
receive a deterministic `Retry-After: 0.001` header; read-timeout retries retain
the SDK's actual backoff. Response scripts are deterministic; scheduler timings
are not. The server waits for request threads during teardown.

The following single samples measure the extraction call through successful
parsing. They exclude client/server construction and teardown, include any
first-use import/setup inside the extraction, and are **not** provider latency
benchmarks or evidence of a speedup. `[200, 200]` means malformed JSON followed
by valid JSON; the four-status sequence has malformed JSON in its first 200.
All rows use two SDK retries and two allowed Instructor validation retries.

| SDK | Execution | HTTP status sequence | Instructor attempts | HTTP requests | ms / successful extraction |
|---|---|---|---:|---:|---:|
| openai | sync | `[200]` | 1 | 1 | 347.9 |
| openai | async | `[200]` | 1 | 1 | 24.7 |
| openai | sync | `[429, 200]` | 1 | 2 | 7.3 |
| openai | async | `[429, 200]` | 1 | 2 | 7.0 |
| openai | sync | `[500, 200]` | 1 | 2 | 7.5 |
| openai | async | `[500, 200]` | 1 | 2 | 7.3 |
| openai | sync | `[200, 200]` | 2 | 2 | 5.2 |
| openai | async | `[200, 200]` | 2 | 2 | 6.1 |
| openai | sync | `[429, 200, 500, 200]` | 2 | 4 | 8.3 |
| openai | async | `[429, 200, 500, 200]` | 2 | 4 | 9.7 |
| anthropic | sync | `[200]` | 1 | 1 | 15.9 |
| anthropic | async | `[200]` | 1 | 1 | 3.0 |
| anthropic | sync | `[429, 200]` | 1 | 2 | 4.3 |
| anthropic | async | `[429, 200]` | 1 | 2 | 5.1 |
| anthropic | sync | `[500, 200]` | 1 | 2 | 3.9 |
| anthropic | async | `[500, 200]` | 1 | 2 | 8.7 |
| anthropic | sync | `[200, 200]` | 2 | 2 | 2.6 |
| anthropic | async | `[200, 200]` | 2 | 2 | 6.4 |
| anthropic | sync | `[429, 200, 500, 200]` | 2 | 4 | 8.4 |
| anthropic | async | `[429, 200, 500, 200]` | 2 | 4 | 8.1 |

Failure measurements are not assigned a time per successful extraction because
there is no successful extraction:

| Script / settings | Instructor attempts | HTTP requests | Result |
|---|---:|---:|---|
| Persistent 429; SDK retries=2 | 1 | 3 | SDK RateLimitError wrapped |
| Persistent 500; SDK retries=0 | 1 | 1 | SDK InternalServerError wrapped |
| 400; SDK retries=2 | 1 | 1 | SDK BadRequestError wrapped |
| Invalid field type; Instructor retries=2 | 3 | 3 | Validation exhaustion |
| Delayed response; SDK retries=0 / 1 | 1 | 1 / 2 | SDK APITimeoutError wrapped |

Both SDKs and both execution modes produced these counts. SDK retries resend
the same request body; validation retries append correction messages. No extra
unrequested HTTP retries were observed under the default integer policy.
`completion:kwargs` counts SDK invocations, `completion:response` counts returned
SDK responses, and `parse:error` counts failed parses. None counts internal SDK
HTTP retries. Exception `n_attempts` counts Instructor attempts. API failures
remain separate from parsing-failure history.

## Timeout and cancellation evidence

A 429 carrying `Retry-After: 0.15`, followed by an immediate successful HTTP
response, exceeds numeric `timeout=0.05`. A valid extraction is still returned;
an invalid extraction stops after that first Instructor attempt. This measures
that the numeric elapsed stop condition does not interrupt SDK retry backoff.
The test asserts elapsed time is at least 150 ms; it imposes no machine-speed
upper threshold.

A response delayed by 300 ms with a 50 ms timeout produces the SDK's
`APITimeoutError`. One configured SDK retry makes two HTTP requests even though
Instructor reports one attempt. Separately, explicit task cancellation and
`asyncio.wait_for` propagate cancellation during an in-flight request, with one
HTTP request and no completion-error or terminal-failure hook. These tests await
the cancelled task and close SDK/server resources; they do not claim cancellation
stops remote computation or provides a hard wall-clock bound during cleanup.

Source inspection additionally establishes: timeout objects and SDK-only
client timeout settings do not supply Instructor's numeric stop condition;
custom Tenacity instances retain their own stop policy. Lazy stream consumption
occurs after the retry wrapper returns. These distinctions are documented but
not separately measured in this test matrix. A parse-error hook can still have
`is_last_attempt=False` when elapsed stopping subsequently terminates the loop;
use the terminal hook rather than predicting policy decisions from error hooks.

## Verification and scope limits

```sh
uv sync --frozen --all-extras
uv run --frozen pytest -q -s tests/core/test_retry_http_semantics.py tests/v2/test_retry_runtime.py tests/v2/test_retry_budget.py tests/coverage/test_core_patch_retry_coverage.py
uv run --frozen ruff check instructor tests
uv run --frozen ruff format --check instructor tests
uv run --frozen ty check instructor/ tests/core/test_retry_http_semantics.py --error-on-warning
```

Final focused run: **127 passed in 10.06 seconds**, including 64 local HTTP cases.
Repository-wide lint and format checks, full-package types plus the new test
file, and whitespace checks passed in the all-extras environment. The first
type-check attempt with only dev extras could not resolve optional SDK imports;
installing the locked extras resolved it without changing dependencies.

The current official [OpenAI retry/timeout documentation](https://github.com/openai/openai-python#retries)
and [Anthropic SDK documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python)
were consulted alongside the installed SDK implementations. Current upstream
OpenAI documentation describes HTTPX2 and current Anthropic documentation
covers a newer major version; measurements here apply to the lockfile versions,
not those untested versions. Other providers, OpenAI Responses, streaming,
connection/DNS failures, alternate HTTP transports, and production latency were
not measured. No full live-provider suite or security conclusion is claimed.

The user guide also fixes examples that put SDK/validation exception predicates
outside Instructor (where exhaustion is already wrapped), and removes an
unsupported `retry_delay` constructor argument. New examples were syntax checked;
no documentation example was sent to a paid API.
