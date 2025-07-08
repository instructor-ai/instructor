#!/usr/bin/env python3
"""Test OpenAI batch validation with simulated successful response"""

import json
import sys
import os
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


def create_simulated_openai_response() -> str:
    """Create a simulated successful OpenAI batch response"""
    
    # Expected results
    users = [
        {"name": "Alice", "age": 28},
        {"name": "Bob", "age": 35},
        {"name": "Sarah", "age": 42},
        {"name": "Mike", "age": 31},
    ]
    
    responses = []
    for i, user in enumerate(users):
        response = {
            "id": f"batch_req_test_{i}",
            "custom_id": f"request-{i}",
            "response": {
                "status_code": 200,
                "request_id": f"test_request_{i}",
                "body": {
                    "id": f"chatcmpl-test{i}",
                    "object": "chat.completion",
                    "created": 1751943000,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(user)
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 20,
                        "total_tokens": 70
                    }
                }
            },
            "error": None
        }
        responses.append(json.dumps(response))
    
    return '\n'.join(responses)


def test_openai_parsing():
    """Test parsing of OpenAI batch results"""
    print("🧪 Testing OpenAI Batch Result Parsing")
    print("=" * 45)
    
    # Create processor
    processor = BatchProcessor("openai/gpt-4o-mini", User)
    
    # Generate simulated response
    simulated_response = create_simulated_openai_response()
    print("📄 Created simulated OpenAI batch response")
    
    # Parse results
    successful_results, error_results = processor.parse_results(simulated_response)
    
    print(f"✅ Successful extractions: {len(successful_results)}")
    print(f"❌ Failed extractions: {len(error_results)}")
    
    # Expected results
    expected_results = [
        User(name="Alice", age=28),
        User(name="Bob", age=35),
        User(name="Sarah", age=42),
        User(name="Mike", age=31),
    ]
    
    print("\n🔍 Validation Results:")
    print("-" * 25)
    
    if len(successful_results) != len(expected_results):
        print(f"❌ Expected {len(expected_results)} results, got {len(successful_results)}")
        return False
    
    # Sort both lists by name for comparison
    results_sorted = sorted(successful_results, key=lambda x: x.name)
    expected_sorted = sorted(expected_results, key=lambda x: x.name)
    
    all_correct = True
    for i, (actual, expected) in enumerate(zip(results_sorted, expected_sorted)):
        if actual.name == expected.name and actual.age == expected.age:
            print(f"✅ {i+1}. {actual.name}, age {actual.age} - CORRECT")
        else:
            print(f"❌ {i+1}. Expected: {expected.name}, age {expected.age}")
            print(f"    Got: {actual.name}, age {actual.age}")
            all_correct = False
    
    if all_correct:
        print(f"\n🎉 All OpenAI extractions are correct!")
        return True
    else:
        print(f"\n⚠️  Some OpenAI extractions have errors")
        return False


if __name__ == "__main__":
    success = test_openai_parsing()
    print(f"\n{'✅ TEST PASSED' if success else '❌ TEST FAILED'}")