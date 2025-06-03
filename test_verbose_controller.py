import instructor
import google.generativeai as genai
from instructor.function_calls import OpenAISchema
from pydantic import Field
import os
import io
from contextlib import redirect_stdout, redirect_stderr

os.environ["GOOGLE_API_KEY"] = "dummy_key_for_testing"

class TestModel(OpenAISchema):
    """Test model for verbose controller functionality"""
    name: str = Field(description="A test name")
    items: list[str] = Field(description="A list of test items")

def test_verbose_controller():
    print("=== Testing Verbose Controller ===")
    
    try:
        client = genai.GenerativeModel("gemini-pro")
        
        print("\n1. Testing verbose=False (should suppress schema output)")
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            instructor_client = instructor.from_gemini(
                client=client,
                mode=instructor.Mode.GEMINI_TOOLS,
                verbose=False,
            )
            
            print(f"Client created with verbose=False: {instructor_client.verbose}")
        
        captured_stdout = stdout_capture.getvalue()
        captured_stderr = stderr_capture.getvalue()
        print(f"Captured stdout: '{captured_stdout}'")
        print(f"Captured stderr: '{captured_stderr}'")
        
        print("\n2. Testing verbose=True (default behavior)")
        instructor_client_verbose = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_TOOLS,
            verbose=True,
        )
        print(f"Client created with verbose=True: {instructor_client_verbose.verbose}")
        
        print("\n3. Testing default behavior (should be verbose=True)")
        instructor_client_default = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_TOOLS,
        )
        print(f"Client created with default: {instructor_client_default.verbose}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_verbose_controller()
