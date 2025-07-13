import os
import pytest

# This will cause all tests in this directory to be skipped if XAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
