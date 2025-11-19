# conftest.py
import os
import pytest
import importlib.util


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring ANTHROPIC_API_KEY"
    )


@pytest.fixture(autouse=True)
def check_api_key_requirement(request):
    """Skip tests marked with 'requires_api_key' if API key is not set."""
    if request.node.get_closest_marker("requires_api_key"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY environment variable not set")

        if importlib.util.find_spec("anthropic") is None:
            pytest.skip("anthropic package is not installed")
