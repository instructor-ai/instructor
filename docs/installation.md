# Installation

=== "pip"
    ```bash
    pip install instructor
    ```

=== "uv"
    ```bash
    uv pip install instructor
    ```

## Version Compatibility

Instructor supports both Pydantic v1 and v2. The library automatically detects which version you have installed and adjusts its behavior accordingly.

=== "Pydantic v1"
    ```python
    from pydantic import BaseModel

    class UserProfile(BaseModel):
        name: str
        age: int

    # Pydantic v1 style validation
    profile = UserProfile(**{"name": "John", "age": 30})
    ```

=== "Pydantic v2"
    ```python
    from pydantic import BaseModel

    class UserProfile(BaseModel):
        name: str
        age: int

    # Pydantic v2 style validation
    profile = UserProfile.model_validate({"name": "John", "age": 30})
    ```

Both versions work seamlessly with Instructor. The library automatically detects your Pydantic version and uses the appropriate validation method.

## Development Installation

For development, you can install additional dependencies:

=== "pip"
    ```bash
    pip install -e ".[dev]"
    ```

=== "uv"
    ```bash
    uv pip install -e ".[dev]"
    ```

This will install all the necessary dependencies for development, including testing and documentation tools.
