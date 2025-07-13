# conftest.py
import os
import pytest

if not os.getenv("COHERE_API_KEY"):
    pytest.skip(
        "COHERE_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from cohere import Client, AsyncClient
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("cohere package is not installed", allow_module_level=True)


@pytest.fixture(scope="session")
def client():
    yield Client()


@pytest.fixture(scope="session")
def aclient():
    yield AsyncClient()
