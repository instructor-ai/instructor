import os
import pytest

if not os.getenv("PINSTRIPES_API_KEY"):
    pytest.skip(
        "PINSTRIPES_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("openai package is not installed", allow_module_level=True)

PINSTRIPES_BASE_URL = "https://pinstripes.io/v1"


@pytest.fixture(scope="function")
def client():
    yield OpenAI(
        api_key=os.environ["PINSTRIPES_API_KEY"],
        base_url=PINSTRIPES_BASE_URL,
    )


@pytest.fixture(scope="function")
def aclient():
    yield AsyncOpenAI(
        api_key=os.environ["PINSTRIPES_API_KEY"],
        base_url=PINSTRIPES_BASE_URL,
    )
