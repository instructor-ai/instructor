# conftest.py
from google.genai import Client
import pytest
import os


@pytest.fixture(scope="function")
def client():
    yield Client()


@pytest.fixture(scope="function")
def aclient():
    yield Client()
