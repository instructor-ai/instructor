from math import e
import pytest
from pytest_examples import find_examples, CodeExample, EvalExample


@pytest.mark.parametrize("example", find_examples("README.md"), ids=str)
def test_readme(example: CodeExample, eval_example: EvalExample):
    eval_example.format(example)


@pytest.mark.parametrize("example", find_examples("docs/index.md"), ids=str)
def test_index(example: CodeExample, eval_example: EvalExample):
    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
        eval_example.run_print_check(example)


@pytest.mark.skip("This is a test for the blog post, which is often broken up")
@pytest.mark.parametrize("example", find_examples("docs/blog/posts"), ids=str)
def test_format_blog(example: CodeExample, eval_example: EvalExample):
    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
        eval_example.run_print_check(example)


@pytest.mark.parametrize("example", find_examples("docs/concepts"), ids=str)
def test_format_concepts(example: CodeExample, eval_example: EvalExample):
    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
        eval_example.run_print_check(example)


@pytest.mark.skip(
    "This is a test for the examples posts which often have many broken up examples"
)
@pytest.mark.parametrize("example", find_examples("docs/examples"), ids=str)
def test_format_examples(example: CodeExample, eval_example: EvalExample):
    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
        eval_example.run_print_check(example)
