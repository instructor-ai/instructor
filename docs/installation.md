---
title: Installing Instructor with Pip
description: Learn how to install Instructor and its dependencies using pip for Python 3.9+. Simple setup guide included.
---

Installation is as simple as:

```bash
pip install instructor
```

Instructor has a few core dependencies:

- [`openai`](https://pypi.org/project/openai/): OpenAI's Python client.
- [`pydantic`](https://pypi.org/project/pydantic/): Data validation and settings management using python type annotations.
- [`docstring-parser`](https://pypi.org/project/docstring-parser/): A parser for Python docstrings, to improve the experience of working with docstrings in jsonschema.

## Optional Dependencies

### CLI Dependencies

If you want to use the Instructor CLI tools, install with the CLI extras:

```bash
pip install "instructor[cli]"
```

This will install additional dependencies:

- [`typer`](https://pypi.org/project/typer/): Build great CLIs. Easy to code. Based on Python type hints.
- [`rich`](https://pypi.org/project/rich/): Rich text and beautiful formatting in the terminal.
- [`aiohttp`](https://pypi.org/project/aiohttp/): Async HTTP client/server framework.

### Provider-specific Dependencies

To use specific LLM providers, you can install the required dependencies:

```bash
# For Anthropic Claude
pip install "instructor[anthropic]"

# For Gemini
pip install "instructor[google-generativeai]"

# For multiple providers
pip install "instructor[anthropic,google-generativeai,cli]"
```

If you've got Python 3.9+ and `pip` installed, you're good to go.
