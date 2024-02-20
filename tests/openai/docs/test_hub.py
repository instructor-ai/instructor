import pytest
from pytest_examples import find_examples, CodeExample, EvalExample


@pytest.mark.parametrize("example", find_examples("docs/hub"), ids=str)
def test_format_blog(example: CodeExample, eval_example: EvalExample):
    if "ollama" in example.source:
        return

    if "llama_cpp" in example.source:
        return

    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
        eval_example.run(example)
