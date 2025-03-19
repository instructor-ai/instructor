# Contributing to Instructor

Thank you for considering contributing to Instructor! This document provides guidelines and instructions to help you contribute effectively.

## Table of Contents

- [Contributing to Instructor](#contributing-to-instructor)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [Getting Started](#getting-started)
    - [Environment Setup](#environment-setup)
    - [Development Workflow](#development-workflow)
    - [Dependency Management](#dependency-management)
      - [Using UV](#using-uv)
      - [Using Poetry](#using-poetry)
    - [Working with Optional Dependencies](#working-with-optional-dependencies)
  - [How to Contribute](#how-to-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Feature Requests](#feature-requests)
    - [Pull Requests](#pull-requests)
    - [Writing Documentation](#writing-documentation)
    - [Contributing to Evals](#contributing-to-evals)
  - [Code Style Guidelines](#code-style-guidelines)
    - [Conventional Comments](#conventional-comments)
    - [Conventional Commits](#conventional-commits)
      - [Types](#types)
      - [Examples](#examples)
  - [Testing](#testing)
  - [Branch and Release Process](#branch-and-release-process)
  - [Using Cursor for PR Creation](#using-cursor-for-pr-creation)
  - [License](#license)

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct: treat everyone with respect, be constructive in your communication, and focus on the technical aspects of the contributions.

## Getting Started

### Environment Setup

1. **Fork the Repository**: Click the "Fork" button at the top right of the [repository page](https://github.com/instructor-ai/instructor).

2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/instructor.git
   cd instructor
   ```

3. **Set up Remote**:
   ```bash
   git remote add upstream https://github.com/instructor-ai/instructor.git
   ```

4. **Install UV** (recommended):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows PowerShell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

5. **Install Dependencies**:
   ```bash
   # Using uv (recommended)
   uv pip install -e ".[dev,docs,test-docs]"
   
   # Using poetry
   poetry install --with dev,docs,test-docs
   
   # For specific providers, add the provider name as an extra
   # Example: uv pip install -e ".[dev,docs,test-docs,anthropic]"
   ```

6. **Set up Pre-commit**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Development Workflow

1. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes and Commit**:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

3. **Keep Your Branch Updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Push Changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Dependency Management

We support both UV and Poetry for dependency management. Choose the tool that works best for you:

#### Using UV

UV is a fast Python package installer and resolver. It's recommended for day-to-day development in Instructor.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project and development dependencies
uv pip install -e ".[dev,docs]"

# Adding a new dependency (example)
uv pip install new-package
```

Key UV commands:
- `uv pip install -e .` - Install the project in editable mode
- `uv pip install -e ".[dev]"` - Install with development extras
- `uv pip freeze > requirements.txt` - Generate requirements file
- `uv self update` - Update UV to the latest version

#### Using Poetry

Poetry provides more comprehensive dependency management and packaging.

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies including development deps
poetry install --with dev,docs

# Add a new dependency
poetry add package-name

# Add a new development dependency
poetry add --group dev package-name
```

Key Poetry commands:
- `poetry shell` - Activate the virtual environment
- `poetry run python -m pytest` - Run commands within the virtual environment
- `poetry update` - Update dependencies to their latest versions

### Working with Optional Dependencies

Instructor uses optional dependencies to support different LLM providers. When adding integration for a new provider:

1. **Update pyproject.toml**: Add your provider's dependencies to both `[project.optional-dependencies]` and `[dependency-groups]`:

   ```toml
   [project.optional-dependencies]
   # Add your provider here
   my-provider = ["my-provider-sdk>=1.0.0,<2.0.0"]
   
   [dependency-groups]
   # Also add to dependency groups
   my-provider = ["my-provider-sdk>=1.0.0,<2.0.0"]
   ```

2. **Create Provider Client**: Implement your provider client in `instructor/clients/client_myprovider.py`

3. **Add Tests**: Create tests in `tests/llm/test_myprovider/`

4. **Document Installation**: Update the documentation to include installation instructions:
   ```
   # Install with your provider support
   uv pip install "instructor[my-provider]"
   # or
   poetry install --with my-provider
   ```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on [our issue tracker](https://github.com/instructor-ai/instructor/issues) with:

1. A clear, descriptive title
2. A detailed description including:
   - The `response_model` you are using
   - The `messages` you are using
   - The `model` you are using
   - Steps to reproduce the bug
   - The expected behavior and what went wrong
   - Your environment (Python version, OS, package versions)

### Feature Requests

For feature requests, please create an issue describing:

1. The problem your feature would solve
2. How your solution would work
3. Alternatives you've considered
4. Examples of how the feature would be used

### Pull Requests

1. **Create a Pull Request** from your fork to the main repository.
2. **Fill out the PR template** with details about your changes.
3. **Address review feedback** and make requested changes.
4. **Wait for CI checks** to pass.
5. Once approved, a maintainer will merge your PR.

### Writing Documentation

Documentation improvements are always welcome! Follow these guidelines:

1. Documentation is written in Markdown format in the `docs/` directory
2. When creating new markdown files, add them to `mkdocs.yml` under the appropriate section
3. Follow the existing hierarchy and structure
4. Use a grade 10 reading level (simple, clear language)
5. Include working code examples
6. Add links to related documentation

### Contributing to Evals

We encourage contributions to our evaluation tests:

1. Explore existing evals in the [evals directory](https://github.com/instructor-ai/instructor/tree/main/tests/llm/test_openai/evals)
2. Contribute new evals as pytest tests
3. Evals should test specific capabilities or edge cases of the library or models
4. Follow the existing patterns for structuring eval tests

## Code Style Guidelines

We use automated tools to maintain consistent code style:

- **Ruff**: For linting and formatting
- **PyRight**: For type checking
- **Black**: For code formatting (enforced by Ruff)

General guidelines:

- **Typing**: Use strict typing with annotations for all functions and variables
- **Imports**: Standard lib → third-party → local imports
- **Models**: Define structured outputs as Pydantic BaseModel subclasses
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use custom exceptions from exceptions.py, validate with Pydantic
- **Comments**: Docstrings for public functions, inline comments for complex logic

### Conventional Comments

We use conventional comments in code reviews and commit messages. This helps make feedback clearer and more actionable:

```
<label>: <subject>

<description>
```

Labels include:
- **praise:** highlights something positive
- **suggestion:** proposes a change or improvement
- **question:** asks for clarification
- **nitpick:** minor, trivial feedback that can be ignored
- **issue:** points out a specific problem that needs to be fixed
- **todo:** notes something to be addressed later
- **fix:** resolves an issue
- **refactor:** suggests reorganizing code without changing behavior
- **test:** suggests adding or improving tests

Examples:
```
suggestion: consider using Pydantic's validator for this check
This would ensure validation happens automatically when the model is created.

question: why is this approach used instead of async processing?
I'm wondering if there would be performance benefits.

fix: correct the type hint for the client parameter
The client should accept OpenAI instances, not strings.
```

For more details, see the [Conventional Comments specification](https://conventionalcomments.org/).

### Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages. This helps us generate changelogs and understand the changes at a glance.

The commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to our CI configuration files and scripts

#### Examples

```
feat(openai): add support for response_format parameter

fix(anthropic): correct tool calling format in Claude client

docs: improve installation instructions for various providers

test(evals): add evaluation for recursive schema handling
```

Breaking changes should be indicated by adding `!` after the type/scope:

```
feat(api)!: change parameter order in from_openai factory function
```

Including a scope is recommended when changes affect a specific part of the codebase (e.g., a specific provider, feature, or component).

## Testing

Run tests using pytest:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/path_to_test.py::test_name

# Skip LLM tests (faster for local development)
pytest tests/ -k 'not llm and not openai'

# Generate coverage report
coverage run -m pytest tests/ -k "not docs"
coverage report
```

## Branch and Release Process

- `main` branch is the development branch
- Releases are tagged with version numbers
- We follow [Semantic Versioning](https://semver.org/)

## Using Cursor for PR Creation

Cursor (https://cursor.sh) is a code editor powered by AI that can help you create PRs efficiently. We encourage using Cursor for Instructor development:

1. **Install Cursor**: Download from [cursor.sh](https://cursor.sh/)

2. **Create a Branch**: Start a new branch for your feature using Cursor's Git integration

3. **Use Cursor Rules**: We have Cursor rules that help with standards:
   - `new-features-planning`: Use when implementing new features
   - `simple-language`: Follow when writing documentation
   - `documentation-sync`: Reference when making code changes to keep docs in sync

4. **Generate Code with AI**: Use Cursor's AI assistance to generate code that follows our style

5. **Auto-Create PRs**: Use Cursor's PR creation feature with our template:
   ```
   # Create PR using gh CLI
   gh pr create -t "Your PR Title" -b "Description of changes" -r jxnl,ivanleomk
   ```

6. **Include Attribution**: Add `This PR was written by [Cursor](https://cursor.sh)` to your PR description

For more details, see our Cursor rules in `.cursor/rules/`.

## License

By contributing to Instructor, you agree that your contributions will be licensed under the project's MIT License. 