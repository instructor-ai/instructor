import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_writer():
    api_key = os.getenv("WRITER_API_KEY")
    if not api_key:
        pytest.skip("WRITER_API_KEY environment variable not set")
