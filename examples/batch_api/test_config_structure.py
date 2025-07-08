#!/usr/bin/env python3
"""Test to understand GenerateContentConfig structure"""

import inspect
from google.genai import types

print("GenerateContentConfig structure:")
print("=" * 50)

# Check fields
if hasattr(types.GenerateContentConfig, '__annotations__'):
    print("\nAnnotations:")
    for field, type_hint in types.GenerateContentConfig.__annotations__.items():
        print(f"  {field}: {type_hint}")

# Try to create config
try:
    config = types.GenerateContentConfig(
        generation_config={
            'temperature': 0.5,
            'max_output_tokens': 100
        }
    )
    print("\nGenerateContentConfig with generation_config created successfully")
except Exception as e:
    print(f"\nError with generation_config: {e}")

# Try direct fields
try:
    config = types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens=100
    )
    print("\nGenerateContentConfig with direct fields created successfully")
    print("Config dict:", config.__dict__ if hasattr(config, '__dict__') else 'No dict')
except Exception as e:
    print(f"\nError with direct fields: {e}")