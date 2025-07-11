#!/usr/bin/env python3
"""
Local test script to reproduce the issue from #1690.
This script demonstrates the fix for GenAI response_model=None.
"""

import instructor
from pydantic import BaseModel

# Example from the issue
model_name = "gemini-2.5-flash-lite-preview-06-17"
messages = [
    {
        "role": "user",
        "content": "What is the capital of France?"
    },
]

class MyModel(BaseModel):
    answer: str

def test_with_response_model():
    """Test that worked before"""
    print("Testing with response_model...")
    
    client = instructor.from_provider(f"google/{model_name}")
    result = client.chat.completions.create(
        messages=messages,
        response_model=MyModel
    )
    print(f"Result: {result}")

def test_without_response_model():
    """Test that was broken before the fix"""
    print("Testing with response_model=None...")
    
    client = instructor.from_provider(f"google/{model_name}")
    result = client.chat.completions.create(
        messages=messages,
        response_model=None
    )
    print(f"Result: {result}")

if __name__ == "__main__":
    print("Running GenAI response_model=None fix test...")
    
    try:
        test_with_response_model()
        print("✓ response_model test passed")
    except Exception as e:
        print(f"✗ response_model test failed: {e}")
    
    try:
        test_without_response_model()
        print("✓ response_model=None test passed")
    except Exception as e:
        print(f"✗ response_model=None test failed: {e}")