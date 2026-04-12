"""Benchmark import time and memory usage for instructor.

Usage:
    python benchmarks/import_benchmark.py
"""

import subprocess
import sys


def measure(label: str, stmt: str) -> None:
    code = f"""
import sys, time, tracemalloc
from collections import Counter

tracemalloc.start()
before = set(sys.modules.keys())
start = time.time()
{stmt}
elapsed = time.time() - start
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

new = set(sys.modules.keys()) - before
print(f"{{elapsed:.3f}}s  {{peak / 1024 / 1024:6.1f}}MB  {{len(new):>5}} modules")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    out = result.stdout.strip()
    print(f"  {label:<35} {out}")
    if result.returncode != 0:
        err = result.stderr.strip().split("\n")[-1]
        print(f"    ERROR: {err}")


def main() -> None:
    print("Import benchmark")
    print("=" * 70)
    print(f"  {'target':<35} {'time':>7}  {'peak':>7}  {'modules':>7}")
    print("-" * 70)

    measure("import instructor", "import instructor")
    measure("instructor + from_openai()", (
        "import instructor; "
        "instructor.from_openai(None)"
    ))
    measure("instructor + from_anthropic()", (
        "import instructor; "
        "import anthropic; "
        "instructor.from_anthropic(anthropic.Anthropic(api_key='test'))"
    ))
    measure("import openai (baseline)", "import openai")
    measure("import anthropic (baseline)", "import anthropic")


if __name__ == "__main__":
    main()
