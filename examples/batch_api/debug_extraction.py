#!/usr/bin/env python3
"""Debug the extraction process step by step"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from instructor.batch import BatchProcessor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Test data from the actual Anthropic response
test_data = {
    "custom_id": "request-0",
    "result": {
        "message": {
            "content": [
                {
                    "input": {
                        "name": "Alice",
                        "age": 28
                    },
                    "name": "extract_data",
                    "type": "tool_use"
                }
            ]
        },
        "type": "succeeded"
    }
}

print("Testing Anthropic extraction...")
print(f"Input data: {json.dumps(test_data, indent=2)}")

processor = BatchProcessor("anthropic/claude-3-5-sonnet-20241022", User)

# Test the _extract_from_response method directly
extracted_data = processor._extract_from_response(test_data)
print(f"\nExtracted data: {extracted_data}")

if extracted_data:
    try:
        user = User(**extracted_data)
        print(f"✅ Successfully created User: {user}")
        print(f"   Type: {type(user)}")
        print(f"   Name: {user.name}")
        print(f"   Age: {user.age}")
    except Exception as e:
        print(f"❌ Failed to create User: {e}")
else:
    print("❌ No data extracted")

# Test the full parsing process
test_content = json.dumps(test_data)
print(f"\n=== FULL PARSING TEST ===")
all_results = processor.parse_results(test_content)
print(f"Total results: {len(all_results)}")
for result in all_results:
    if result.success:
        print(f"✅ Success ({result.custom_id}): {result.result}")
    else:
        print(f"❌ Error ({result.custom_id}): {result.error_message}")
        print(f"   Raw data: {result.raw_data}")