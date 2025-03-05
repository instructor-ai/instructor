import os

import pytest
from openai import OpenAI


@pytest.fixture(scope="session")
def client():
    if os.environ.get("PERPLEXITY_API_KEY"):
        yield OpenAI(
            api_key=os.environ["PERPLEXITY_API_KEY"],
            base_url="https://api.perplexity.ai",
        )
