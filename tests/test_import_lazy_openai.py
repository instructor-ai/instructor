"""Regression test for issue #2205: import instructor should not eagerly load openai.

Verifies that accessing core instructor symbols (Instructor, from_provider, Mode)
does not trigger the openai SDK import, which is expensive (~30MB, ~550 modules).
The openai SDK should only load when a user explicitly creates an OpenAI-based client.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _check_openai_not_loaded(access_code: str) -> None:
    """Run a subprocess that accesses an instructor symbol and asserts openai is not loaded."""
    script = textwrap.dedent(f"""
        import sys
        import instructor
        {access_code}
        if "openai" in sys.modules:
            loaded = [m for m in sys.modules if m.startswith("openai")]
            print(f"FAIL: openai loaded ({{len(loaded)}} modules)")
            sys.exit(1)
        else:
            print("PASS: openai not loaded")
            sys.exit(0)
    """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"openai was eagerly loaded after: {access_code}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_bare_import_does_not_load_openai():
    _check_openai_not_loaded("pass")


def test_from_provider_access_does_not_load_openai():
    _check_openai_not_loaded("_ = instructor.from_provider")


def test_instructor_class_access_does_not_load_openai():
    _check_openai_not_loaded("_ = instructor.Instructor")


def test_mode_access_does_not_load_openai():
    _check_openai_not_loaded("_ = instructor.Mode")


def test_partial_access_does_not_load_openai():
    _check_openai_not_loaded("_ = instructor.Partial")


@pytest.mark.parametrize("symbol", [
    "from_provider",
    "Instructor",
    "AsyncInstructor",
    "Mode",
    "Partial",
    "Maybe",
])
def test_core_symbols_accessible_without_openai(symbol: str):
    """Core symbols must remain accessible (no import errors)."""
    script = textwrap.dedent(f"""
        import instructor
        obj = getattr(instructor, "{symbol}")
        assert obj is not None, f"{{symbol}} resolved to None"
    """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Failed to access instructor.{symbol}\n"
        f"stderr: {result.stderr}"
    )
