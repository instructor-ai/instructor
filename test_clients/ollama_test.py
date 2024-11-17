import sys
import os
import openai
import instructor
from pydantic import BaseModel
from typing import List, Optional

def test_ollama_basic():
    print("Testing Ollama basic functionality...")

    # Configure OpenAI client with Ollama endpoint
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    # Enable instructor patches with JSON mode
    client = instructor.patch(client, mode=instructor.Mode.JSON)

    class User(BaseModel):
        name: str
        age: int

    try:
        # Test basic extraction
        user = client.chat.completions.create(
            model="llama2",
            messages=[
                {"role": "user", "content": "Extract: Jason is 25 years old"},
            ],
            response_model=User,
        )
        print(f"Basic test result: {user}")
        return True
    except Exception as e:
        print(f"Error in basic test: {str(e)}")
        if "connection refused" in str(e).lower():
            print("Error: Ollama server not running. Please start with 'ollama serve'")
        elif "model not found" in str(e).lower():
            print("Error: Model not available. Run 'ollama pull llama2'")
        return False

def test_ollama_nested():
    print("\nTesting Ollama nested objects...")

    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    client = instructor.patch(client, mode=instructor.Mode.JSON)

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class User(BaseModel):
        name: str
        age: int
        addresses: List[Address]

    try:
        user = client.chat.completions.create(
            model="llama2",
            messages=[
                {"role": "user", "content": """
                    Extract: Jason is 25 years old.
                    He lives at 123 Main St, New York, USA
                    and has a summer house at 456 Beach Rd, Miami, USA
                """},
            ],
            response_model=User,
        )
        print(f"Nested test result: {user}")
        return True
    except Exception as e:
        print(f"Error in nested test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Ollama integration tests...")
    basic_success = test_ollama_basic()
    nested_success = test_ollama_nested()

    if basic_success and nested_success:
        print("\nAll tests passed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please check the error messages above.")
        sys.exit(1)
