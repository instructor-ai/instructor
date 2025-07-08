#!/usr/bin/env python3
"""Show the actual parsed Pydantic objects from batch results"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from instructor.batch import BatchProcessor, filter_successful, filter_errors, extract_results
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

def show_anthropic_results():
    """Show parsed Anthropic batch results"""
    print("🧠 ANTHROPIC BATCH RESULTS")
    print("=" * 50)
    
    # Read batch ID
    with open("anthropic_batch_id.txt", "r") as f:
        batch_id = f.read().strip()
    
    print(f"Batch ID: {batch_id}")
    
    # Get results using BatchProcessor
    processor = BatchProcessor("anthropic/claude-3-5-sonnet-20241022", User)
    all_results = processor.retrieve_results(batch_id)
    
    print(f"\nTotal results: {len(all_results)}")
    
    # Show each result with detailed info
    for i, result in enumerate(all_results):
        print(f"\n--- Result {i+1} ---")
        print(f"Custom ID: {result.custom_id}")
        print(f"Success: {result.success}")
        
        if result.success:
            user = result.result
            print(f"✅ PARSED USER OBJECT:")
            print(f"   Type: {type(user)}")
            print(f"   Name: {user.name}")
            print(f"   Age: {user.age}")
            print(f"   JSON: {user.model_dump_json()}")
            print(f"   Dict: {user.model_dump()}")
            
            # Test that it's a real Pydantic object
            print(f"   Is BaseModel: {isinstance(user, BaseModel)}")
            print(f"   Is User: {isinstance(user, User)}")
            
            # Test Pydantic methods
            try:
                validated = User.model_validate(user.model_dump())
                print(f"   Re-validation: ✅ Works")
                print(f"   Re-validated: {validated}")
            except Exception as e:
                print(f"   Re-validation: ❌ {e}")
        else:
            print(f"❌ ERROR:")
            print(f"   Type: {result.error_type}")
            print(f"   Message: {result.error_message}")
    
    # Test the utility functions
    successful_results = filter_successful(all_results)
    error_results = filter_errors(all_results)
    extracted_users = extract_results(all_results)
    
    print(f"\n🔍 UTILITY FUNCTIONS:")
    print(f"Successful results: {len(successful_results)}")
    print(f"Error results: {len(error_results)}")
    print(f"Extracted users: {len(extracted_users)}")
    
    if extracted_users:
        print(f"\n📋 EXTRACTED USER OBJECTS:")
        for user in extracted_users:
            print(f"  • {user.name}, age {user.age} (type: {type(user).__name__})")

def show_openai_results():
    """Show parsed OpenAI batch results"""
    print("\n🤖 OPENAI BATCH RESULTS")
    print("=" * 50)
    
    # Read batch ID
    with open("openai_batch_id.txt", "r") as f:
        batch_id = f.read().strip()
    
    print(f"Batch ID: {batch_id}")
    
    # Get results using BatchProcessor
    processor = BatchProcessor("openai/gpt-4o-mini", User)
    
    # Check status first
    status = processor.get_batch_status(batch_id)
    print(f"Status: {status['status']}")
    
    if status['status'] == 'completed':
        all_results = processor.retrieve_results(batch_id)
        
        print(f"\nTotal results: {len(all_results)}")
        
        # Show each result with detailed info
        for i, result in enumerate(all_results):
            print(f"\n--- Result {i+1} ---")
            print(f"Custom ID: {result.custom_id}")
            print(f"Success: {result.success}")
            
            if result.success:
                user = result.result
                print(f"✅ PARSED USER OBJECT:")
                print(f"   Type: {type(user)}")
                print(f"   Name: {user.name}")
                print(f"   Age: {user.age}")
                print(f"   JSON: {user.model_dump_json()}")
                print(f"   Dict: {user.model_dump()}")
                
                # Test that it's a real Pydantic object
                print(f"   Is BaseModel: {isinstance(user, BaseModel)}")
                print(f"   Is User: {isinstance(user, User)}")
            else:
                print(f"❌ ERROR:")
                print(f"   Type: {result.error_type}")
                print(f"   Message: {result.error_message}")
        
        # Test the utility functions
        extracted_users = extract_results(all_results)
        if extracted_users:
            print(f"\n📋 EXTRACTED USER OBJECTS:")
            for user in extracted_users:
                print(f"  • {user.name}, age {user.age} (type: {type(user).__name__})")
    else:
        print(f"Batch not completed yet: {status['status']}")

if __name__ == "__main__":
    show_anthropic_results()
    show_openai_results()