# conftest.py
from openai import AsyncOpenAI, OpenAI
import pytest
import os

try:
    import braintrust

    wrap_openai = braintrust.wrap_openai
except ImportError:

    def wrap_openai(x):
        return x


@pytest.fixture(scope="session")
def client():
    if os.environ.get("BRAINTRUST_API_KEY"):
        yield wrap_openai(
            OpenAI(
                api_key=os.environ["BRAINTRUST_API_KEY"],
                base_url="https://braintrustproxy.com/v1",
            )
        )
    elif (
        os.environ.get("OPENAI_BASE_URL", None)
        == "https://api.endpoints.anyscale.com/v1"
    ):
        yield OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.endpoints.anyscale.com/v1",
        )

    else:
        yield OpenAI()


@pytest.fixture(scope="session")
def aclient():
    if os.environ.get("BRAINTRUST_API_KEY"):
        yield wrap_openai(
            AsyncOpenAI(
                api_key=os.environ["BRAINTRUST_API_KEY"],
                base_url="https://braintrustproxy.com/v1",
            )
        )
    elif (
        os.environ.get("OPENAI_BASE_URL", None)
        == "https://api.endpoints.anyscale.com/v1"
    ):
        yield AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.endpoints.anyscale.com/v1",
        )
    else:
        yield AsyncOpenAI()
