# conftest.py
import os
import pytest

if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("BRAINTRUST_API_KEY")):
    pytest.skip(
        "ANTHROPIC_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from anthropic import AsyncAnthropic, Anthropic
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("anthropic package is not installed", allow_module_level=True)

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
