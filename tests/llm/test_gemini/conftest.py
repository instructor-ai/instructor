# conftest.py

import pytest
import os


import google.generativeai as genai
from functools import lru_cache


@pytest.fixture(scope="session", autouse=True)
def setup_gemini():
    print("Setting up Gemini")
    # Check if GOOGLE_API_KEY is set in environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set in environment variables")

    # Configure the Gemini API
    genai.configure(api_key=api_key)
