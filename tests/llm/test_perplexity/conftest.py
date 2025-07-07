import os

import pytest

if not os.getenv("PERPLEXITY_API_KEY"):
    pytest.skip(
        "PERPLEXITY_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("openai package is not installed", allow_module_level=True)


@pytest.fixture(scope="session")
def client():
    yield OpenAI(
        api_key=os.environ["PERPLEXITY_API_KEY"],
        base_url="https://api.perplexity.ai",
    )
