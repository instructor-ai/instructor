#!/usr/bin/env python3
"""
Debug script to understand CI environment differences.
This will help identify why thinking_config works locally but fails in CI.
"""

import sys
import os

def check_environment():
    """Check environment details that might affect thinking_config validation."""
    print("=== Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    env_vars = ["INSTRUCTOR_ENV", "GOOGLE_API_KEY", "UV_CACHE_DIR"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if var == "GOOGLE_API_KEY":
                print(f"{var}: {'***' if value else 'Not set'}")
            else:
                print(f"{var}: {value}")
        else:
            print(f"{var}: Not set")
    
    try:
        import google.genai
        print(f"google-genai version: {google.genai.__version__}")
    except ImportError as e:
        print(f"google-genai import failed: {e}")
    
    try:
        import pydantic
        print(f"pydantic version: {pydantic.__version__}")
    except ImportError as e:
        print(f"pydantic import failed: {e}")

def test_thinking_config_validation_edge_cases():
    """Test various thinking_config validation scenarios."""
    print("\n=== Testing thinking_config Validation Edge Cases ===")
    
    try:
        from google.genai import types
        from instructor.providers.gemini.utils import update_genai_kwargs
        
        test_cases = [
            {"thinking_budget": 0},
            {"thinking_budget": 1024},
            {"thinkingBudget": 0},
            {"thinkingBudget": 1024},
            None,
        ]
        
        for i, thinking_config in enumerate(test_cases, 1):
            print(f"{i}. Testing thinking_config: {thinking_config}")
            
            try:
                config_direct = types.GenerateContentConfig(thinking_config=thinking_config)
                print(f"   ✓ Direct creation: {config_direct.thinking_config}")
            except Exception as e:
                print(f"   ✗ Direct creation failed: {e}")
            
            try:
                new_kwargs = {'thinking_config': thinking_config} if thinking_config is not None else {}
                base_config = {}
                result = update_genai_kwargs(new_kwargs, base_config)
                config_utils = types.GenerateContentConfig(**result)
                print(f"   ✓ Through utils: {config_utils.thinking_config}")
            except Exception as e:
                print(f"   ✗ Through utils failed: {e}")
                
    except Exception as e:
        print(f"✗ Edge case testing failed: {e}")

def reproduce_exact_ci_failure():
    """Reproduce the exact CI failure scenario."""
    print("\n=== Reproducing Exact CI Failure ===")
    
    os.environ["INSTRUCTOR_ENV"] = "CI"
    
    try:
        from instructor.providers.gemini.utils import update_genai_kwargs
        from google.genai import types
        
        test_cases = [
            {"thinking_budget": 0},
            None
        ]
        
        for thinking_config in test_cases:
            print(f"\nTesting thinking_config: {thinking_config}")
            
            new_kwargs = {
                'messages': [{'content': 'Jason is 25 years old, Alice is 30 years old', 'role': 'user'}], 
                'model': 'gemini-2.5-flash', 
                'stream': True
            }
            
            if thinking_config is not None:
                new_kwargs['thinking_config'] = thinking_config
            
            base_config = {
                "system_instruction": None,
                "tools": [types.Tool(function_declarations=[])],
                "tool_config": types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY", allowed_function_names=["PartialIterableUserExtract"]
                    )
                ),
            }
            
            print("1. Calling update_genai_kwargs...")
            generation_config = update_genai_kwargs(new_kwargs, base_config)
            print(f"   Result keys: {list(generation_config.keys())}")
            print(f"   thinking_config: {generation_config.get('thinking_config')}")
            
            print("2. Creating GenerateContentConfig...")
            try:
                config = types.GenerateContentConfig(**generation_config)
                print(f"   ✓ Success: {config.thinking_config}")
            except Exception as e:
                print(f"   ✗ Failed: {e}")
                print(f"   Error type: {type(e)}")
                
                print("   Analyzing generation_config:")
                for key, value in generation_config.items():
                    print(f"     {key}: {type(value)} = {repr(value)}")
                    
                return False
                
    except Exception as e:
        print(f"✗ CI reproduction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    check_environment()
    test_thinking_config_validation_edge_cases()
    success = reproduce_exact_ci_failure()
    
    if success:
        print("\n✓ All tests passed - issue may be environment-specific")
    else:
        print("\n✗ Reproduced the CI failure locally")
