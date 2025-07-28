#!/usr/bin/env python3
"""
Debug script to understand thinking_config validation issues.
"""

def test_thinking_config_formats():
    """Test different thinking_config formats to understand what works."""
    
    print("=== Testing unit test format (snake_case) ===")
    from instructor.providers.gemini.utils import update_genai_kwargs
    
    thinking_config_snake = {"thinking_budget": 1024}
    kwargs = {"thinking_config": thinking_config_snake}
    base_config = {}
    
    try:
        result = update_genai_kwargs(kwargs, base_config)
        print(f"✓ update_genai_kwargs with snake_case: {result.get('thinking_config')}")
    except Exception as e:
        print(f"✗ update_genai_kwargs with snake_case failed: {e}")
    
    print("\n=== Testing API format (camelCase) ===")
    thinking_config_camel = {"thinkingBudget": 1024}
    kwargs = {"thinking_config": thinking_config_camel}
    base_config = {}
    
    try:
        result = update_genai_kwargs(kwargs, base_config)
        print(f"✓ update_genai_kwargs with camelCase: {result.get('thinking_config')}")
    except Exception as e:
        print(f"✗ update_genai_kwargs with camelCase failed: {e}")
    
    print("\n=== Testing GenerateContentConfig creation ===")
    try:
        from google.genai import types
        
        try:
            config_snake = types.GenerateContentConfig(thinking_config=thinking_config_snake)
            print(f"✓ GenerateContentConfig with snake_case: {config_snake}")
        except Exception as e:
            print(f"✗ GenerateContentConfig with snake_case failed: {e}")
        
        try:
            config_camel = types.GenerateContentConfig(thinking_config=thinking_config_camel)
            print(f"✓ GenerateContentConfig with camelCase: {config_camel}")
        except Exception as e:
            print(f"✗ GenerateContentConfig with camelCase failed: {e}")
            
    except ImportError as e:
        print(f"✗ Cannot import google.genai.types: {e}")

if __name__ == "__main__":
    test_thinking_config_formats()
