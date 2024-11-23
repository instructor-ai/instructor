# Contributing to Instructor

Thank you for your interest in contributing to Instructor! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/your-username/instructor.git
cd instructor
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install development dependencies using UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e ".[test-docs,anthropic,groq,cohere,mistralai,litellm,google-generativeai,vertexai,cerebras_cloud_sdk,fireworks-ai]"
uv pip install -r requirements-dev.txt
```

## Running Tests

To run the tests:
```bash
pytest tests/
```

To run specific test categories:
```bash
pytest tests/ -k 'not llm and not openai and not gemini and not anthropic and not cohere and not vertexai'
```

## Code Style

We use:
- [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- [Pyright](https://github.com/microsoft/pyright) for type checking
- [Black](https://github.com/psf/black) for code formatting

Before submitting a PR, please ensure your code passes all style checks:
```bash
ruff check .
pyright
black .
```

## Documentation

- Update documentation in the `docs/` directory for any new features
- Include docstrings for new functions and classes
- Add examples in `examples/` directory when appropriate

## Pull Request Process

1. Create a new branch for your feature/fix:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them with clear, descriptive messages

3. Push to your fork and submit a pull request

4. Ensure all checks pass in the PR

## Questions?

Feel free to open an issue for any questions about contributing!
