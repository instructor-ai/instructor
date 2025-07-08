import pytest
from pytest_examples import find_examples, CodeExample, EvalExample
import glob
import os

exclusions = ["ollama.md", "watsonx.md", "local_classification.md"]

markdown_files = [
    file
    for file in glob.glob("docs/examples/*.md")
    if os.path.basename(file) not in exclusions
]

code_examples = []

for markdown_file in markdown_files:
    code_examples.extend(find_examples(markdown_file))


@pytest.mark.parametrize("example", code_examples, ids=str)
def test_index(example: CodeExample, eval_example: EvalExample):
    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
