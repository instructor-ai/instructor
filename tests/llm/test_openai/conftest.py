# conftest.py
import os
import pytest

if not (os.getenv("OPENAI_API_KEY") or os.getenv("BRAINTRUST_API_KEY")):
    pytest.skip(
        "OPENAI_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("openai package is not installed", allow_module_level=True)

try:
    import braintrust

    wrap_openai = braintrust.wrap_openai
except ImportError:

    def wrap_openai(x):
        return x


@pytest.fixture(scope="function")
def client():
    if os.environ.get("BRAINTRUST_API_KEY"):
        yield wrap_openai(
            OpenAI(
                api_key=os.environ["BRAINTRUST_API_KEY"],
                base_url="https://braintrustproxy.com/v1",
            )
        )
    else:
        yield OpenAI()


@pytest.fixture(scope="function")
def aclient():
    if os.environ.get("BRAINTRUST_API_KEY"):
        yield wrap_openai(
            AsyncOpenAI(
                api_key=os.environ["BRAINTRUST_API_KEY"],
                base_url="https://braintrustproxy.com/v1",
            )
        )
    else:
        yield AsyncOpenAI()
