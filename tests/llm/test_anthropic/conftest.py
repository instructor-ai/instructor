# conftest.py
from anthropic import AsyncAnthropic, Anthropic
import pytest
import os

try:
    import braintrust

    wrap_anthropic = braintrust.wrap_anthropic
except ImportError:

    def wrap_anthropic(x):
        return x


@pytest.fixture(scope="session")
def client():
    if os.environ.get("BRAINTRUST_API_KEY"):
        yield wrap_anthropic(
            Anthropic(
                api_key=os.environ["BRAINTRUST_API_KEY"],
                base_url="https://braintrustproxy.com/v1",
            )
        )
    else:
        yield Anthropic()


@pytest.fixture(scope="session")
def aclient():
    if os.environ.get("BRAINTRUST_API_KEY"):
        yield wrap_anthropic(
            AsyncAnthropic(
                api_key=os.environ["BRAINTRUST_API_KEY"],
                base_url="https://braintrustproxy.com/v1",
            )
        )
    else:
        yield AsyncAnthropic()
