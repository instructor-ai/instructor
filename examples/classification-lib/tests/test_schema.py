import os
import sys

# Adding parent directory to path to import schema
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import ClassificationDefinition


def test_load_intent_classification_yaml():
    # Get the path to the test YAML file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")

    # Load the YAML file
    classification_def = ClassificationDefinition.from_yaml(yaml_path)

    # Verify the loaded data
    assert classification_def.system_message.startswith(
        "You are an expert classification system"
    ), "Unexpected system_message content"
    assert len(classification_def.label_definitions) == 3

    # Check that labels are lowercase as per the validator
    labels = [ld.label for ld in classification_def.label_definitions]
    assert "question" in labels
    assert "scheduling" in labels
    assert "coding" in labels

    # Check examples for a specific label
    question_label = next(
        ld for ld in classification_def.label_definitions if ld.label == "question"
    )
    assert len(question_label.examples.examples_positive) == 4
    assert len(question_label.examples.examples_negative) == 4
    assert "What is machine learning?" in question_label.examples.examples_positive
