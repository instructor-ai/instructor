# conftest.py
from openai import AsyncOpenAI, OpenAI
import pytest


try:
    import braintrust

    wrap_openai = braintrust.wrap_openai
except ImportError:

    def wrap_openai(x):
        return x


@pytest.fixture(scope="session")
def client():
    yield OpenAI()


@pytest.fixture(scope="session")
def aclient():
    yield AsyncOpenAI()
