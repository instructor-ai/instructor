# Schema Tests

This directory contains tests for the classification schema validation functionality.

## Running Tests

To run all tests:

```bash
pytest -xvs tests/
```

## Test Files

- `test_schema.py` - Tests for schema validation (lowercase labels and unique labels)
- `test_examples.py` - Tests for the Examples class functionality

## Adding New Tests

When adding new tests:

1. Create a new test file with the `test_` prefix
2. Import the necessary classes from the schema module
3. Write test functions with the `test_` prefix
4. Run the tests to verify functionality 