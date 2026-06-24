"""Benchmark script for instructor import performance.

Measures memory usage and module count at various import stages to identify
bottlenecks in the import chain. Run with:

    python scripts/benchmark_import.py

Requires: tracemalloc (stdlib), optionally the provider SDKs for full profiling.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def run_import_benchmark(code: str, label: str) -> dict[str, str]:
    """Run a Python snippet in a subprocess and capture memory/timing stats."""
    script = textwrap.dedent(f"""
        import tracemalloc
        import time
        import sys

        tracemalloc.start()
        start = time.time()

        {code}

        elapsed = time.time() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mod_count = len(sys.modules)
        print(f"{{elapsed:.3f}}|{{current}}|{{peak}}|{{mod_count}}")
    """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        return {
            "label": label,
            "time": "ERROR",
            "memory_mb": "ERROR",
            "peak_mb": "ERROR",
            "modules": "ERROR",
            "error": result.stderr.strip(),
        }

    parts = result.stdout.strip().split("|")
    elapsed, current, peak, mod_count = parts

    return {
        "label": label,
        "time": f"{float(elapsed):.3f}s",
        "memory_mb": f"{int(current) / 1024 / 1024:.1f} MB",
        "peak_mb": f"{int(peak) / 1024 / 1024:.1f} MB",
        "modules": mod_count,
    }


def main() -> None:
    print("=" * 70)
    print(" INSTRUCTOR IMPORT BENCHMARK")
    print("=" * 70)
    print(f" Python: {sys.version.split()[0]}")
    print(f" Executable: {sys.executable}")
    print("=" * 70)
    print()

    benchmarks = [
        ("import sys", "baseline (sys only)"),
        ("import instructor", "import instructor"),
        ("import instructor; _ = instructor.from_provider", "access from_provider"),
        ("import instructor; _ = instructor.from_openai", "access from_openai"),
        ("import instructor; _ = instructor.from_anthropic", "access from_anthropic"),
        ("import instructor; _ = instructor.from_genai", "access from_genai"),
        ("import instructor; _ = instructor.Mode", "access Mode"),
        ("import instructor; _ = instructor.Instructor", "access Instructor class"),
        ("import instructor; _ = instructor.Partial", "access Partial"),
    ]

    results = []
    for code, label in benchmarks:
        result = run_import_benchmark(code, label)
        results.append(result)

    header = f"{'Benchmark':<30} {'Time':>8} {'Memory':>10} {'Peak':>10} {'Modules':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r.get("error"):
            print(f"{r['label']:<30} {'ERROR':>8} (missing dependency?)")
        else:
            print(
                f"{r['label']:<30} {r['time']:>8} {r['memory_mb']:>10} "
                f"{r['peak_mb']:>10} {r['modules']:>8}"
            )

    print()
    print("Notes:")
    print("  - 'import instructor' should be <1MB and <100 modules (lazy loading)")
    print("  - Provider access loads the SDK on demand (expected ~20-50MB)")
    print("  - If 'import instructor' alone is >50MB, lazy loading is broken")


if __name__ == "__main__":
    main()
