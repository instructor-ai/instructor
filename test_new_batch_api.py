#!/usr/bin/env python3
"""Test script for the new Maybe-like batch API design"""

from instructor.batch import (
    BatchProcessor, 
    filter_successful, 
    filter_errors, 
    extract_results,
    get_results_by_custom_id
)
from pydantic import BaseModel
from typing import List

class User(BaseModel):
    name: str
    age: int

def test_new_api():
    """Test the new Maybe-like API design"""
    print("🧪 Testing new Maybe-like Batch API design")
    
    # Wait for the latest batch to process
    batch_id = "msgbatch_01RaCPc8CjmsLDyGu2W1LhEw"
    
    # Use our processor
    processor = BatchProcessor("anthropic/claude-3-5-sonnet-20241022", User)
    
    # Get batch status first
    status = processor.get_batch_status(batch_id)
    print(f"📊 Batch Status: {status['status']}")
    
    if status['status'] != "ended":
        print(f"⏳ Batch is {status['status']}, testing with mock data instead")
        # Create some mock results for testing
        mock_results = [
            processor.parse_results('{"custom_id": "test-1", "result": {"type": "message", "message": {"content": [{"type": "tool_use", "input": {"name": "Alice", "age": 28}}]}}}')[0],
        ]
    else:
        # Get real results
        print("🔄 Retrieving batch results...")
        mock_results = processor.retrieve_results(batch_id)
    
    print(f"\n📊 Results Summary:")
    print(f"Total results: {len(mock_results)}")
    
    # Filter results using the new helper functions
    successful_results = filter_successful(mock_results)
    error_results = filter_errors(mock_results)
    extracted_users = extract_results(mock_results)
    results_by_id = get_results_by_custom_id(mock_results)
    
    print(f"✅ Successful: {len(successful_results)}")
    print(f"❌ Errors: {len(error_results)}")
    
    # Show successful results
    for success in successful_results:
        print(f"   ✅ {success.custom_id}: {success.result.name}, age {success.result.age}")
    
    # Show errors with detailed information
    for error in error_results:
        print(f"   ❌ {error.custom_id}: {error.error_type} - {error.error_message}")
    
    # Test accessing by custom_id
    print(f"\n🔍 Access by custom_id:")
    for custom_id, result in results_by_id.items():
        if result.success:
            print(f"   {custom_id}: SUCCESS - {result.result}")
        else:
            print(f"   {custom_id}: ERROR - {result.error_message}")
    
    print(f"\n🎯 Just the extracted users: {len(extracted_users)}")
    for user in extracted_users:
        print(f"   - {user.name}, age {user.age}")

if __name__ == "__main__":
    test_new_api()
