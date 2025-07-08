#!/usr/bin/env python3
"""Test that batch parsing returns proper Pydantic objects"""

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
    
    def describe(self) -> str:
        """Example method to verify we have a real Pydantic object"""
        return f"{self.name} is {self.age} years old"


def test_pydantic_parsing():
    """Test that BatchProcessor returns proper Pydantic objects"""
    print("🧪 Testing Pydantic Object Parsing")
    print("=" * 40)
    
    # Create simulated response data
    response_data = {
        "id": "batch_req_test_1",
        "custom_id": "request-0",
        "response": {
            "status_code": 200,
            "request_id": "test_request_1",
            "body": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"name": "Alice", "age": 28})
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
        }
    }
    
    simulated_content = json.dumps(response_data)
    print("📄 Created simulated batch response")
    
    # Test BatchProcessor parsing
    processor = BatchProcessor("openai/gpt-4o-mini", User)
    successful_results, error_results = processor.parse_results(simulated_content)
    
    print(f"✅ Successful extractions: {len(successful_results)}")
    print(f"❌ Failed extractions: {len(error_results)}")
    
    if successful_results:
        user = successful_results[0]
        
        # Verify it's a proper Pydantic object
        print(f"\n🔍 Type Verification:")
        print(f"  • Type: {type(user)}")
        print(f"  • Is BaseModel: {isinstance(user, BaseModel)}")
        print(f"  • Is User: {isinstance(user, User)}")
        
        # Test Pydantic features
        print(f"\n🎯 Pydantic Features:")
        print(f"  • user.name: {user.name}")
        print(f"  • user.age: {user.age}")
        print(f"  • user.describe(): {user.describe()}")
        print(f"  • user.model_dump(): {user.model_dump()}")
        print(f"  • user.model_dump_json(): {user.model_dump_json()}")
        
        # Verify it's not just a dict
        try:
            # This should work for Pydantic objects
            validated_user = User.model_validate(user.model_dump())
            print(f"  • Re-validation works: ✅")
        except Exception as e:
            print(f"  • Re-validation failed: ❌ {e}")
        
        print(f"\n🎉 Successfully parsed into proper Pydantic User object!")
        return True
    else:
        print(f"\n❌ No successful results found")
        return False


if __name__ == "__main__":
    success = test_pydantic_parsing()
    print(f"\n{'✅ TEST PASSED' if success else '❌ TEST FAILED'}")