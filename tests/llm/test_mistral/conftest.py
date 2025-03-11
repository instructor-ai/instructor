# conftest.py
import pytest
import os
from mistralai import Mistral


@pytest.fixture(scope="function")
def client():
    yield Mistral(api_key=os.environ["MISTRAL_API_KEY"])


@pytest.fixture(scope="function")
def aclient():
    yield Mistral(api_key=os.environ["MISTRAL_API_KEY"])
