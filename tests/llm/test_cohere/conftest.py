# conftest.py
from cohere import Client, AsyncClient
import pytest


@pytest.fixture(scope="session")
def client():
    yield Client()


@pytest.fixture(scope="session")
def aclient():
    yield AsyncClient()
