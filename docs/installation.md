---
title: Installing Instructor
description: Learn how to install Instructor and its dependencies using pip or uv for Python 3.9+. Simple setup guide included.
---

# Installation

=== "pip"
    ```bash
    pip install instructor
    ```

=== "uv"
    ```bash
    uv pip install instructor
    ```

Instructor has a few dependencies:

- [`openai`](https://pypi.org/project/openai/): OpenAI's Python client.
- [`typer`](https://pypi.org/project/typer/): Build great CLIs. Easy to code. Based on Python type hints.
- [`docstring-parser`](https://pypi.org/project/docstring-parser/): A parser for Python docstrings, to improve the experience of working with docstrings in jsonschema.
- [`pydantic`](https://pypi.org/project/pydantic/): Data validation and settings management using python type annotations.

If you've got Python 3.9+ and either `pip` or `uv` installed, you're good to go.

!!! tip "Using UV"
    [UV](https://github.com/astral-sh/uv) is a new, extremely fast Python package installer and resolver. It's a great alternative to pip, offering significantly faster installation times.
