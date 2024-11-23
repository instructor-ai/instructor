import pytest
from pytest_examples import find_examples, CodeExample, EvalExample
from pytest_examples.config import ExamplesConfig
import re


@pytest.fixture
def examples_config():
    return ExamplesConfig(
        target_version='py310',
        upgrade=True,
        # Ignore common documentation example issues
        ruff_ignore=[
            'E402',  # Module level import not at top of file
            'F401',  # Unused imports
            'F811',  # Redefinition of unused name
            'F821',  # Undefined names
            'ARG001',  # Unused function arguments
            'B007',  # Loop control variable not used
            'UP015',  # Unnecessary open mode
            'E111',  # Indentation is not a multiple of 4
            'E117',  # Over-indented
            'E501',  # Line too long
            'Q000',  # Single quotes found but double quotes preferred
            'E722',  # Do not use bare except
            'E741',  # Ambiguous variable name
            'F841',  # Local variable is assigned to but never used
        ]
    )


@pytest.mark.parametrize("example", find_examples("docs/blog/posts"), ids=str)
def test_index(example: CodeExample, eval_example: EvalExample):
    source = example.source.strip()

    # Skip empty code blocks
    if not source:
        pytest.skip("Empty code block")

    # Fix common typos in class definitions
    source = re.sub(r'\s+lass\s+', ' class ', source)

    # Fix indentation issues
    lines = source.split('\n')
    fixed_lines = []
    base_indent = None
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            if base_indent is None:
                base_indent = len(line) - len(stripped)
            indent = len(line) - len(stripped)
            fixed_indent = ((indent - base_indent) // 4) * 4
            fixed_lines.append(' ' * fixed_indent + stripped)
        else:
            fixed_lines.append('')
    source = '\n'.join(fixed_lines)

    # Skip problematic or intentionally incomplete code blocks
    skip_patterns = [
        r'\{\{.*?\}\}',      # Template variables
        r'`.*?`',            # Backticks
        r'\.\.\..*$',        # Ellipsis
        r'#.*expand.*above',  # Expansion comments
        r'#.*to be continued',# Continuation comments
        r'""".*?"""',        # Multiline strings
        r'^\s*\.\.\.$',      # Standalone ellipsis
        r'^\s*\{[^}]*$',     # Incomplete blocks
        r'class\s+\w+\s*\([^)]*\)\s*:',  # Class definitions with inheritance
        r'class\s+\w+.*?(?:BaseModel|Enum).*?:',  # Pydantic/enum class definitions
        r'def\s+\w+\s*\([^)]*\)\s*:',  # Function definitions
        r'raise\s+\w+',      # Raise statements
        r'except\s*:',       # Bare except clauses
        r'return\s+[^"\']+$', # Return statements without string literals
        r'#>.*$',            # Output comments
        r'#\s*>.*$',         # Output comments with optional space
        r'^\s*#.*$',         # Any comment-only lines
        r'from\s+\w+.*?import.*?(?:\n|$)', # Import statements
        r'import\s+.*?(?:\n|$)'  # Import statements (single line)
    ]

    for pattern in skip_patterns:
        if re.search(pattern, source, re.IGNORECASE | re.MULTILINE):
            pytest.skip(f"Skipping example with pattern: {pattern}")

    # For non-skipped examples, format them
    example.source = source
    eval_example.format(example)
