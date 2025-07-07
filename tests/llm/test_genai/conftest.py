# conftest.py
import os
import pytest

if not os.getenv("GOOGLE_API_KEY"):
    pytest.skip(
        "GOOGLE_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from google.genai import Client
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("google-genai package is not installed", allow_module_level=True)


@pytest.fixture(scope="function")
def client():
    yield Client()


@pytest.fixture(scope="function")
def aclient():
    yield Client()
