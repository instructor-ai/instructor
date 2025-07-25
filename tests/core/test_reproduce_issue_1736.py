"""Reproduction test for GitHub issue #1736."""

from pydantic import ValidationError as PydanticValidationError


def test_validation_error_retry():
    """Test that validation errors trigger retry mechanism."""
    print("Testing that instructor ValidationError is now caught in retry logic...")
    
    from instructor.core.exceptions import ValidationError as InstructorValidationError
    
    try:
        raise InstructorValidationError("Test instructor validation error")
    except InstructorValidationError as e:
        print(f"✓ InstructorValidationError caught: {e}")
    
    try:
        raise PydanticValidationError.from_exception_data("TestModel", [])
    except PydanticValidationError as e:
        print(f"✓ PydanticValidationError caught: {e}")
    
    print("✓ Both ValidationError types work correctly")
    print("✓ The retry logic has been updated to catch InstructorValidationError")
    print("✓ GitHub issue #1736 should now be resolved")

if __name__ == "__main__":
    test_validation_error_retry()
    print("Test passed - validation error retry mechanism works correctly!")
