---
title: Contribute to Instructor: Evals, Issues, and Pull Requests
description: Join us in enhancing the Instructor library with evals, report issues, and submit pull requests on GitHub. Collaborate and contribute!
---

# Contributing to Instructor

We welcome contributions to Instructor! This page covers the different ways you can help improve the library.

## Ways to Contribute

### Evaluation Tests (Evals)

Evals help us monitor the quality of both the OpenAI models and the Instructor library. To contribute:

1. **Explore Existing Evals**: Check out [our evals directory](https://github.com/instructor-ai/instructor/tree/main/tests/llm/test_openai/evals)
2. **Create a New Eval**: Add new pytest tests that evaluate specific capabilities or edge cases
3. **Follow the Pattern**: Structure your eval similar to existing ones
4. **Submit a PR**: We'll review and incorporate your eval

Evals are run weekly, and results are tracked to monitor performance over time.

### Reporting Issues

If you encounter a bug or problem, please [file an issue on GitHub](https://github.com/instructor-ai/instructor/issues) with:

1. A clear, descriptive title
2. Detailed information including:
   - The `response_model` you're using
   - The `messages` you sent
   - The `model` you're using
   - Steps to reproduce the issue
   - Expected vs. actual behavior
   - Your environment details (Python version, OS, package versions)

### Contributing Code

We welcome pull requests! Here's the process:

1. **For Small Changes**: Feel free to submit a PR directly
2. **For Larger Changes**: [Start with an issue](https://github.com/instructor-ai/instructor/issues) to discuss approach
3. **Looking for Ideas?** Check issues labeled [help wanted](https://github.com/instructor-ai/instructor/labels/help%20wanted) or [good first issue](https://github.com/instructor-ai/instructor/labels/good%20first%20issue)

## Setting Up Your Development Environment

### Using UV (Recommended)

UV is a fast Python package installer and resolver that makes development easier.

1. **Install UV** (official method):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows PowerShell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Install Project in Development Mode**:
   ```bash
   # Clone the repository
   git clone https://github.com/YOUR-USERNAME/instructor.git
   cd instructor
   
   # Install with development dependencies 
   uv pip install -e ".[dev,docs]"
   ```

3. **Adding New Dependencies**:
   ```bash
   # Add a regular dependency
   uv pip install some-package
   
   # Install a specific version
   uv pip install "some-package>=1.0.0,<2.0.0"
   ```

4. **Common UV Commands**:
   ```bash
   # Update UV itself
   uv self update
   
   # Create a requirements file
   uv pip freeze > requirements.txt
   ```

### Using Poetry

Poetry provides comprehensive dependency management and packaging.

1. **Install Poetry**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install Dependencies**:
   ```bash
   # Clone the repository
   git clone https://github.com/YOUR-USERNAME/instructor.git
   cd instructor
   
   # Install with development dependencies
   poetry install --with dev,docs
   ```

3. **Working with Poetry**:
   ```bash
   # Activate virtual environment
   poetry shell
   
   # Run a command in the virtual environment
   poetry run pytest
   
   # Add a dependency
   poetry add package-name
   
   # Add a development dependency
   poetry add --group dev package-name
   ```

## Adding Support for New LLM Providers

Instructor uses optional dependencies to support different LLM providers. To add a new provider:

1. **Add Dependencies to pyproject.toml**:
   ```toml
   [project.optional-dependencies]
   # Add your provider
   my-provider = ["my-provider-sdk>=1.0.0,<2.0.0"]
   
   [dependency-groups]
   # Mirror in dependency groups
   my-provider = ["my-provider-sdk>=1.0.0,<2.0.0"]
   ```

2. **Create Provider Client**:
   - Create a new file at `instructor/clients/client_myprovider.py`
   - Implement `from_myprovider` function that patches the provider's client

3. **Add Tests**: Create tests in `tests/llm/test_myprovider/`

4. **Document Installation**:
   ```bash
   # Installation command for your provider
   uv pip install "instructor[my-provider]"
   # or with poetry
   poetry install --with my-provider
   ```

5. **Write Documentation**:
   - Add a new markdown file in `docs/integrations/` for your provider
   - Update `mkdocs.yml` to include your new page
   - Make sure to include a complete example

## Development Workflow

1. **Fork the Repository**: Create your own fork of the project
2. **Clone and Set Up**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/instructor.git
   cd instructor
   git remote add upstream https://github.com/instructor-ai/instructor.git
   ```
3. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make Changes, Test, and Commit**:
   ```bash
   # Run tests
   pytest tests/ -k 'not llm and not openai'  # Skip LLM tests for faster local dev
   
   # Commit changes
   git add .
   git commit -m "Your descriptive commit message"
   ```
5. **Keep Updated and Push**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request**: Submit your PR with a clear description of changes

## Using Cursor to Build PRs

[Cursor](https://cursor.sh) is an AI-powered code editor that can help you contribute to Instructor.

1. **Getting Started with Cursor**:
   - Download Cursor from [cursor.sh](https://cursor.sh)
   - Open the Instructor project in Cursor
   - Cursor will automatically detect our rules in `.cursor/rules/`

2. **Using Cursor Rules**:
   - `new-features-planning`: Helps plan and structure new features
   - `simple-language`: Guidelines for writing clear documentation
   - `documentation-sync`: Ensures documentation stays in sync with code changes

3. **Creating PRs with Cursor**:
   - Use Cursor's Git integration to create a new branch
   - Make your changes with AI assistance
   - Create a PR with:
     ```bash
     # Use GitHub CLI to create the PR
     gh pr create -t "Your feature title" -b "Description of your changes" -r jxnl,ivanleomk
     ```
   - Add `This PR was written by [Cursor](https://cursor.sh)` to your PR description

4. **Benefits of Using Cursor**:
   - AI helps generate code that follows our style guidelines
   - Simplifies PR creation process
   - Helps maintain documentation standards

## Code Style Guidelines

We use the following tools to maintain code quality:

- **Ruff**: For linting and formatting
- **PyRight**: For type checking
- **Pre-commit**: For automatic checks before committing

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

Key style guidelines:
- Use strict typing
- Follow import order: standard lib → third-party → local
- Use snake_case for functions/variables, PascalCase for classes
- Write comprehensive docstrings for public API functions

### Conventional Comments

When reviewing code or writing commit messages, we use conventional comments to make feedback clearer:

```
<label>: <subject>

<description>
```

Common labels:
- **praise:** highlights something positive
- **suggestion:** proposes a change or improvement 
- **question:** asks for clarification
- **issue:** points out a problem that needs fixing
- **todo:** notes something to be addressed later
- **fix:** resolves an issue

Examples:

```
suggestion: use a validator for this field
This would ensure the value is always properly formatted.

question: why not use async processing here?
I'm curious if this would improve performance.

fix: correct the parameter type
It should be an OpenAI client instance, not a string.
```

This format helps everyone understand the purpose and importance of each comment. Visit [conventionalcomments.org](https://conventionalcomments.org/) to learn more.

### Conventional Commits

We use conventional commit messages to make our project history clear and generate automated changelogs. A conventional commit has this structure:

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

#### Common Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Formatting changes
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **test**: Adding or fixing tests
- **chore**: Maintenance tasks

#### Examples

```
feat(openai): add streaming response support

fix(anthropic): resolve tool calling response format

docs: update installation instructions

test(evals): add new recursive schema test cases
```

For breaking changes, add an exclamation mark before the colon:

```
feat(api)!: change return type of from_openai function
```

Using conventional commits helps automatically generate release notes and makes the project history easier to navigate.

For more details, see the [Conventional Commits specification](https://www.conventionalcommits.org/).

## Documentation Contributions

Documentation improvements are highly valued:

1. **Docs Structure**: All documentation is in Markdown in the `docs/` directory
2. **Adding New Pages**: When adding a new page, include it in `mkdocs.yml` in the right section
3. **Local Preview**: Run `mkdocs serve` to preview changes locally
4. **Style Guidelines**:
   - Write at a grade 10 reading level (simple, clear language)
   - Include working code examples
   - Add links to related documentation
   - Use consistent formatting
   - Make sure each code example is complete with imports

Example of a good documentation code block:

```python
# Complete example with imports
import instructor
from openai import OpenAI
from pydantic import BaseModel

# Define your model
class Person(BaseModel):
    name: str
    age: int
    
# Create the patched client
client = instructor.from_openai(OpenAI())

# Use the model
person = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Person,
    messages=[
        {"role": "user", "content": "Extract: John Doe is 25 years old"}
    ]
)

print(person.name)  # "John Doe"
print(person.age)   # 25
```

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/instructor-ai/instructor/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxnl/instructor" />
</a>

## Documentation Resources

When working on documentation, these resources may be helpful:

- **mkdocs serve**: Preview documentation locally. Install dependencies from `requirements-doc.txt` first.

- **hl_lines in Code Blocks**: Highlight specific lines in a code block to draw attention:
  ````markdown
  ```python hl_lines="2 3"
  def example():
      # This line is highlighted
      # This line is also highlighted
      return "normal line"
  ```
  ````

- **Admonitions**: Create styled callout boxes for important information:
  ```markdown
  !!! note "Optional Title"
      This is a note admonition.
  
  !!! warning
      This is a warning.
  ```

For more documentation features, check the [MkDocs Material documentation](https://squidfunk.github.io/mkdocs-material/).

Thank you for your contributions to Instructor!
