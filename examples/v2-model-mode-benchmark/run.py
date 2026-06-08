"""Benchmark live v2 model/mode cells derived from the provider contract.

The default run enumerates every provider with registered v2 handlers and each
of its declared modes. Cells without a configured model, optional SDK, or
credential are reported as skipped instead of failing the run.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import instructor
from pydantic import BaseModel

from instructor.v2.core.errors import ConfigurationError
from instructor.v2.core.mode import Mode
from instructor.v2.core.provider_specs import ALIAS_TO_PROVIDER, PROVIDER_SPECS
from instructor.v2.core.providers import Provider


class Person(BaseModel):
    """Small extraction target used for comparable smoke measurements."""

    name: str
    age: int


PROMPT = "Extract the person from this sentence: Jason is 36 years old."

# API keys needed by model-string builders that can be safely preflighted.
# Cloud credential chains are opt-in through --allow-cloud-auth.
CREDENTIAL_ENV_VARS: dict[Provider, tuple[str, ...]] = {
    Provider.OPENAI: ("OPENAI_API_KEY",),
    Provider.ANYSCALE: ("ANYSCALE_API_KEY",),
    Provider.TOGETHER: ("TOGETHER_API_KEY",),
    Provider.DATABRICKS: ("DATABRICKS_TOKEN", "DATABRICKS_API_KEY"),
    Provider.DEEPSEEK: ("DEEPSEEK_API_KEY",),
    Provider.OPENROUTER: ("OPENROUTER_API_KEY",),
    Provider.ANTHROPIC: ("ANTHROPIC_API_KEY",),
    Provider.GENAI: ("GOOGLE_API_KEY",),
    Provider.GEMINI: ("GOOGLE_API_KEY",),
    Provider.COHERE: ("COHERE_API_KEY",),
    Provider.PERPLEXITY: ("PERPLEXITY_API_KEY",),
    Provider.XAI: ("XAI_API_KEY",),
    Provider.GROQ: ("GROQ_API_KEY",),
    Provider.MISTRAL: ("MISTRAL_API_KEY",),
    Provider.FIREWORKS: ("FIREWORKS_API_KEY",),
    Provider.CEREBRAS: ("CEREBRAS_API_KEY",),
    Provider.WRITER: ("WRITER_API_KEY",),
}


@dataclass(frozen=True)
class BenchmarkCase:
    provider: Provider
    model: str | None
    mode: Mode

    @property
    def label(self) -> str:
        return f"{self.model or self.provider.value + '/<configure-model>'} [{self.mode.name}]"


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    status: str
    successes: int
    trials: int
    latencies_ms: tuple[float, ...] = ()
    detail: str | None = None

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    @property
    def median_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        return statistics.median(self.latencies_ms)

    @property
    def mean_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        return statistics.fmean(self.latencies_ms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.case.provider.value,
            "model": self.case.model,
            "mode": self.case.mode.name,
            "mode_value": self.case.mode.value,
            "status": self.status,
            "successes": self.successes,
            "trials": self.trials,
            "success_rate": self.success_rate,
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "latencies_ms": list(self.latencies_ms),
            "detail": self.detail,
        }


def _benchmark_specs() -> dict[Provider, Any]:
    return {
        provider: spec
        for provider, spec in PROVIDER_SPECS.items()
        if spec.handler_module is not None and spec.supported_modes
    }


def parse_mode(value: str) -> Mode:
    """Accept either the enum name (`TOOLS`) or serialized mode value."""
    try:
        return Mode[value.upper()]
    except KeyError:
        try:
            return Mode(value)
        except ValueError as exc:
            choices = ", ".join(mode.name for mode in Mode)
            raise argparse.ArgumentTypeError(
                f"Unknown mode {value!r}; choose one of {choices}"
            ) from exc


def _provider_for_model(model: str) -> Provider:
    try:
        alias, _ = model.split("/", 1)
    except ValueError as exc:
        raise ValueError(f"Model must be in provider/model form: {model!r}") from exc
    provider = ALIAS_TO_PROVIDER.get(alias)
    if provider is None:
        raise ValueError(f"Unknown provider alias in model: {model!r}")
    spec = PROVIDER_SPECS[provider]
    if spec.handler_module is None or not spec.supported_modes:
        raise ValueError(
            f"Provider alias {alias!r} has no v2 capability contract to benchmark"
        )
    return provider


def build_cases(
    models: Sequence[str] = (),
    modes: Sequence[Mode] = (),
) -> list[BenchmarkCase]:
    """Build a complete provider-mode grid or a grid for explicit models."""
    specs = _benchmark_specs()
    selected: list[tuple[Provider, str | None]]
    if models:
        selected = [(_provider_for_model(model), model) for model in models]
    else:
        selected = [
            (provider, spec.provider_string) for provider, spec in specs.items()
        ]

    cases: list[BenchmarkCase] = []
    for provider, model in selected:
        supported_modes = specs[provider].supported_modes
        selected_modes = tuple(modes) if modes else supported_modes
        cases.extend(
            BenchmarkCase(provider=provider, model=model, mode=mode)
            for mode in selected_modes
        )
    return cases


def _module_available(module_name: str | None) -> bool:
    if module_name is None:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def unavailable_reason(
    case: BenchmarkCase, *, allow_cloud_auth: bool = False
) -> str | None:
    """Return a skip reason before a live request is attempted."""
    spec = PROVIDER_SPECS[case.provider]
    if case.model is None:
        return "no default model configured; pass --model provider/model"
    if case.mode not in spec.supported_modes:
        return "mode is not declared as supported by this provider"
    credentials = CREDENTIAL_ENV_VARS.get(case.provider)
    if credentials is None and not allow_cloud_auth:
        return "provider uses ambient/cloud credentials; pass --allow-cloud-auth"
    if credentials and not any(os.environ.get(name) for name in credentials):
        return f"missing credential: {' or '.join(credentials)}"
    if not _module_available(spec.sdk_module):
        return f"optional SDK is not installed: {spec.sdk_module}"
    return None


def _short_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message[:180]}"


def run_case(
    case: BenchmarkCase,
    trials: int,
    *,
    client_factory: Callable[..., Any] = instructor.from_provider,
    allow_cloud_auth: bool = False,
) -> BenchmarkResult:
    """Run one live cell after preflight checks."""
    reason = unavailable_reason(case, allow_cloud_auth=allow_cloud_auth)
    if reason is not None:
        return BenchmarkResult(case, "skipped", 0, trials, detail=reason)
    assert case.model is not None

    try:
        client = client_factory(case.model, mode=case.mode)
    except (ConfigurationError, ImportError, ModuleNotFoundError) as exc:
        return BenchmarkResult(case, "skipped", 0, trials, detail=_short_error(exc))
    except Exception as exc:
        return BenchmarkResult(case, "failed", 0, trials, detail=_short_error(exc))

    durations: list[float] = []
    successes = 0
    errors: list[str] = []
    for _ in range(trials):
        start = time.perf_counter()
        try:
            response_model = (
                Iterable[Person] if case.mode is Mode.PARALLEL_TOOLS else Person
            )
            result = client.create(
                response_model=response_model,
                messages=[{"role": "user", "content": PROMPT}],
            )
            people = list(result) if case.mode is Mode.PARALLEL_TOOLS else [result]
            if len(people) == 1 and any(
                person.name.strip().lower() == "jason" and person.age == 36
                for person in people
            ):
                successes += 1
            else:
                errors.append(
                    f"unexpected result: {[person.model_dump() for person in people]!r}"
                )
        except Exception as exc:
            errors.append(_short_error(exc))
        durations.append((time.perf_counter() - start) * 1000)

    status = "passed" if successes == trials else "failed"
    detail = "; ".join(errors[:2]) or None
    return BenchmarkResult(
        case=case,
        status=status,
        successes=successes,
        trials=trials,
        latencies_ms=tuple(durations),
        detail=detail,
    )


def run_grid(
    cases: Iterable[BenchmarkCase],
    trials: int,
    *,
    client_factory: Callable[..., Any] = instructor.from_provider,
    allow_cloud_auth: bool = False,
) -> list[BenchmarkResult]:
    return [
        run_case(
            case,
            trials,
            client_factory=client_factory,
            allow_cloud_auth=allow_cloud_auth,
        )
        for case in cases
    ]


def render_markdown(results: Sequence[BenchmarkResult]) -> str:
    """Render stable output that can be checked into a benchmark report."""
    completed = sorted(
        (result for result in results if result.status != "skipped"),
        key=lambda result: (
            -result.success_rate,
            result.median_ms if result.median_ms is not None else float("inf"),
        ),
    )
    rows = ["# Instructor v2 model/mode benchmark", ""]
    if completed:
        rows.extend(
            [
                "## Ranked completed cells",
                "",
                "| Rank | Model | Mode | Success | Median ms |",
                "| ---: | --- | --- | ---: | ---: |",
            ]
        )
        for rank, result in enumerate(completed, start=1):
            median = f"{result.median_ms:.1f}" if result.median_ms is not None else "-"
            rows.append(
                f"| {rank} | `{result.case.model}` | `{result.case.mode.name}` "
                f"| {result.successes}/{result.trials} | {median} |"
            )
        rows.extend(["", "## All cells", ""])
    rows.extend(
        [
            "| Model | Mode | Status | Success | Median ms | Mean ms | Detail |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        median = f"{result.median_ms:.1f}" if result.median_ms is not None else "-"
        mean = f"{result.mean_ms:.1f}" if result.mean_ms is not None else "-"
        detail = (result.detail or "").replace("|", "\\|")
        rows.append(
            f"| `{result.case.model or result.case.provider.value + '/<configure-model>'}` "
            f"| `{result.case.mode.name}` | {result.status} "
            f"| {result.successes}/{result.trials} | {median} | {mean} | {detail} |"
        )
    return "\n".join(rows) + "\n"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("trials must be at least 1")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Limit the grid to a provider/model string; repeat for multiple models.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        type=parse_mode,
        help="Limit each selected model to a mode name or value; repeat as needed.",
    )
    parser.add_argument("--trials", type=_positive_int, default=3)
    parser.add_argument(
        "--allow-cloud-auth",
        action="store_true",
        help="Attempt providers that rely on ambient cloud credentials.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)

    results = run_grid(
        build_cases(models=args.model, modes=args.mode),
        trials=args.trials,
        allow_cloud_auth=args.allow_cloud_auth,
    )
    markdown = render_markdown(results)
    print(markdown, end="")

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps([result.as_dict() for result in results], indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown_out is not None:
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
