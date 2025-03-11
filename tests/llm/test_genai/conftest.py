# conftest.py
from google.genai import Client
import pytest


@pytest.fixture(scope="function")
def client():
    yield Client()


@pytest.fixture(scope="function")
def aclient():
    yield Client()
