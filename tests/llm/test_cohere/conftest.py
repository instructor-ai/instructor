# conftest.py
import os
import pytest

if not os.getenv("COHERE_API_KEY"):
    pytest.skip(
        "COHERE_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from cohere import Client, ClientV2, AsyncClient, AsyncClientV2
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("cohere package is not installed", allow_module_level=True)


@pytest.fixture(scope="session")
def client_v1():
    yield Client()


@pytest.fixture(scope="session")
def client_v2():
    yield ClientV2()


@pytest.fixture(scope="session")
def aclient_v1():
    yield AsyncClient()


@pytest.fixture(scope="session")
def aclient_v2():
    yield AsyncClientV2()
