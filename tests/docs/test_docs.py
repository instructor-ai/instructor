from math import e
import pytest
from pytest_examples import find_examples, CodeExample, EvalExample


@pytest.mark.parametrize("example", find_examples("README.md"), ids=str)
def test_readme(example: CodeExample, eval_example: EvalExample):
    eval_example.format(example)


@pytest.mark.parametrize("example", find_examples("docs/blog/posts"), ids=str)
def test_format_blog(example: CodeExample, eval_example: EvalExample):
    eval_example.format(example)


@pytest.mark.parametrize("example", find_examples("docs/concepts"), ids=str)
def test_format_concepts(example: CodeExample, eval_example: EvalExample):
    eval_example.format(example)
    eval_example.run_print_update(example)


@pytest.mark.parametrize("example", find_examples("docs/examples"), ids=str)
def test_format_examples(example: CodeExample, eval_example: EvalExample):
    eval_example.format(example)
