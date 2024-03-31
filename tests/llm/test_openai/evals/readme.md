# How to Contribute: Writing and Running Evaluation Tests

We welcome contributors to expand our suite of evaluation tests for data extraction. This guide provides instructions on creating tests with `pytest`, `pydantic`, and other tools, focusing on broad coverage and failure modalities understanding.

## Define Test Scenarios

Identify data extraction scenarios relevant to you. Create test cases with inputs and expected outputs.

Reference the `test_extract_users.py` which contains a test case for extracting users, using all models and all modes. The test case is parameterized with the model and mode, and the test function is parameterized with the input and expected output.
