from __future__ import annotations

from openai import OpenAI
from rich.console import Console
import os

from schema import ClassificationDefinition, EvalSet
from classify import Classifier
from eval import evaluate_classifier, display_evaluation_results

# Get current directory to form absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the YAML schema for classification
yaml_path = os.path.join(current_dir, "tests", "intent_classification.yaml")
definition = ClassificationDefinition.from_yaml(yaml_path)

# Load the evaluation set
eval_set_path = os.path.join(current_dir, "tests", "example_evalset.yaml")
eval_set = EvalSet.from_yaml(eval_set_path)

# Initialize the classifier with OpenAI
classifier = (
    Classifier(definition)
    .with_client(OpenAI())
    .with_model("gpt-3.5-turbo")  # Can use gpt-4 for better accuracy
)

# Validate that the eval set is compatible with our definition
if not eval_set.validate_against_definition(definition):
    console = Console()
    console.print(
        "[bold red]Error:[/bold red] Evaluation set contains labels not in the classification definition."
    )
    exit(1)

# Run the evaluation
result = evaluate_classifier(classifier, eval_set, n_jobs=4)

# Display the results (with detailed prediction information)
display_evaluation_results(result, detailed=True)
