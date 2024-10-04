import os
import pytest
from google import generativeai as genai


@pytest.fixture(scope="session", autouse=True)
def configure_genai():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
