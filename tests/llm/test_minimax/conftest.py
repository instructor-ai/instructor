import os

import pytest

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")


@pytest.fixture
def minimax_api_key():
    if not MINIMAX_API_KEY:
        pytest.skip("MINIMAX_API_KEY not set")
    return MINIMAX_API_KEY
