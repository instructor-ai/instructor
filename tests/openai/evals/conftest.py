# conftest.py
from openai import AsyncOpenAI, OpenAI
import pytest


@pytest.fixture(scope="session")
def client():
    yield OpenAI()


@pytest.fixture(scope="session")
def aclient():
    yield AsyncOpenAI()
