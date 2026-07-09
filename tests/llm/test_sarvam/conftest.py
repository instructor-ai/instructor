"""Shared fixtures for Sarvam LLM integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TEST_DIR.parents[2]

if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

_EVALS_DIR = _TEST_DIR / "evals"
if str(_EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALS_DIR))


def _load_dotenv() -> None:
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return
    load_dotenv(env_path)


_load_dotenv()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "sarvam: tests that call the live Sarvam API (requires SARVAM_API_KEY)",
    )


@pytest.fixture(scope="session")
def sarvam_api_key() -> str:
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip(
            "SARVAM_API_KEY is not set. "
            "Add it to .env in the repo root (see .env.example)."
        )
    return api_key
