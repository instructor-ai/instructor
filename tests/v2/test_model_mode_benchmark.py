from __future__ import annotations

from collections.abc import Iterable
import importlib.util
from pathlib import Path
import sys
from typing import Any, get_origin

from instructor.v2.core.mode import Mode
from instructor.v2.core.provider_specs import PROVIDER_SPECS
from instructor.v2.core.providers import Provider


def _load_benchmark() -> Any:
    path = Path(__file__).parents[2] / "examples" / "v2-model-mode-benchmark" / "run.py"
    spec = importlib.util.spec_from_file_location("v2_model_mode_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark: Any = _load_benchmark()


def test_default_grid_includes_every_declared_provider_mode() -> None:
    cases = benchmark.build_cases()
    expected = {
        (provider, mode)
        for provider, spec in PROVIDER_SPECS.items()
        if spec.handler_module is not None
        for mode in spec.supported_modes
    }

    assert {(case.provider, case.mode) for case in cases} == expected


def test_explicit_models_can_compare_multiple_models_in_selected_modes() -> None:
    cases = benchmark.build_cases(
        models=("openai/model-a", "openai/model-b"),
        modes=(Mode.TOOLS, Mode.JSON_SCHEMA),
    )

    assert [(case.model, case.mode) for case in cases] == [
        ("openai/model-a", Mode.TOOLS),
        ("openai/model-a", Mode.JSON_SCHEMA),
        ("openai/model-b", Mode.TOOLS),
        ("openai/model-b", Mode.JSON_SCHEMA),
    ]


def test_missing_key_skips_cell_without_creating_client(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    created = False

    def factory(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        nonlocal created
        created = True
        raise AssertionError("client should not be created")

    result = benchmark.run_case(
        benchmark.BenchmarkCase(Provider.OPENAI, "openai/gpt-4o-mini", Mode.TOOLS),
        trials=1,
        client_factory=factory,
    )

    assert result.status == "skipped"
    assert result.detail == "missing credential: OPENAI_API_KEY"
    assert created is False


def test_successful_cell_records_correctness_and_rendered_latency(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(benchmark, "_module_available", lambda _module: True)

    class FakeClient:
        def create(self, **kwargs: Any) -> Any:  # noqa: ARG002
            return benchmark.Person(name="Jason", age=36)

    def factory(*_args: Any, **_kwargs: Any) -> FakeClient:
        return FakeClient()

    result = benchmark.run_case(
        benchmark.BenchmarkCase(Provider.OPENAI, "openai/gpt-4o-mini", Mode.TOOLS),
        trials=2,
        client_factory=factory,
    )
    rendered = benchmark.render_markdown([result])

    assert result.status == "passed"
    assert result.successes == 2
    assert result.median_ms is not None
    assert "## Ranked completed cells" in rendered
    assert "| `openai/gpt-4o-mini` | `TOOLS` | passed | 2/2 |" in rendered


def test_parallel_cell_uses_iterable_response_model(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(benchmark, "_module_available", lambda _module: True)
    seen_response_model: Any = None

    class FakeClient:
        def create(self, **kwargs: Any) -> Any:
            nonlocal seen_response_model
            seen_response_model = kwargs["response_model"]
            return [benchmark.Person(name="Jason", age=36)]

    result = benchmark.run_case(
        benchmark.BenchmarkCase(
            Provider.OPENAI,
            "openai/gpt-4o-mini",
            Mode.PARALLEL_TOOLS,
        ),
        trials=1,
        client_factory=lambda *_args, **_kwargs: FakeClient(),
    )

    assert result.status == "passed"
    assert get_origin(seen_response_model) is Iterable


def test_cloud_auth_cells_require_explicit_opt_in(monkeypatch: Any) -> None:
    monkeypatch.setattr(benchmark, "_module_available", lambda _module: True)
    case = benchmark.BenchmarkCase(
        Provider.BEDROCK,
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        Mode.TOOLS,
    )

    skipped = benchmark.run_case(case, trials=1)

    assert skipped.status == "skipped"
    assert skipped.detail == (
        "provider uses ambient/cloud credentials; pass --allow-cloud-auth"
    )

    class FakeClient:
        def create(self, **kwargs: Any) -> Any:  # noqa: ARG002
            return benchmark.Person(name="Jason", age=36)

    executed = benchmark.run_case(
        case,
        trials=1,
        allow_cloud_auth=True,
        client_factory=lambda *_args, **_kwargs: FakeClient(),
    )
    assert executed.status == "passed"
