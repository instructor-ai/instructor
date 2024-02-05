import pytest
from pytest_examples import find_examples, CodeExample, EvalExample


@pytest.mark.parametrize("example", find_examples("README.md"), ids=str)
def test_readme(example: CodeExample, eval_example: EvalExample):
    eval_example.format(example)
    eval_example.run_print_update(example)


@pytest.mark.parametrize("example", find_examples("docs/"), ids=str)
def test_readme(example: CodeExample, eval_example: EvalExample):
    eval_example.format(example)
    eval_example.run_print_update(example)
