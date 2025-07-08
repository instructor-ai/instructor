#!/usr/bin/env python3
"""Test the fixed OpenAI schema"""

import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from instructor.batch import BatchRequest
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Create a test batch request
batch_request = BatchRequest[User](
    custom_id="test-0",
    messages=[
        {"role": "system", "content": "Extract user info"},
        {"role": "user", "content": "Hi, I'm Alice, 28 years old"},
    ],
    response_model=User,
    model="gpt-4o-mini",
    max_tokens=100,
    temperature=0.1,
)

# Generate OpenAI format
openai_format = batch_request.to_openai_format()
schema = openai_format["body"]["response_format"]["json_schema"]["schema"]

print("OpenAI JSON Schema:")
print(json.dumps(schema, indent=2))

# Check if additionalProperties is set correctly
def check_additional_properties(schema_dict, path=""):
    """Check if additionalProperties is set to false for all objects"""
    if isinstance(schema_dict, dict):
        if schema_dict.get("type") == "object":
            if "additionalProperties" not in schema_dict:
                print(f"❌ Missing additionalProperties at {path}")
                return False
            elif schema_dict["additionalProperties"] is not False:
                print(f"❌ additionalProperties is not false at {path}")
                return False
            else:
                print(f"✅ additionalProperties is false at {path}")
        
        # Check nested properties
        for key, value in schema_dict.items():
            if isinstance(value, dict):
                if not check_additional_properties(value, f"{path}.{key}"):
                    return False
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        if not check_additional_properties(item, f"{path}.{key}[{i}]"):
                            return False
    
    return True

print("\n=== Schema Validation ===")
if check_additional_properties(schema, "root"):
    print("🎉 Schema is valid for OpenAI strict mode!")
else:
    print("❌ Schema has issues for OpenAI strict mode")