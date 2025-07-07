# conftest.py
import os
import pytest

if not os.getenv("MISTRAL_API_KEY"):
    pytest.skip(
        "MISTRAL_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from mistralai import Mistral
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("mistralai package is not installed", allow_module_level=True)


@pytest.fixture(scope="function")
def client():
    yield Mistral(api_key=os.environ["MISTRAL_API_KEY"])


@pytest.fixture(scope="function")
def aclient():
    yield Mistral(api_key=os.environ["MISTRAL_API_KEY"])
