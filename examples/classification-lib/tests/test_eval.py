import os
import sys
from pydantic import BaseModel

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from schema import EvalSet, ClassificationDefinition
from eval import evaluate_classifier, display_evaluation_results


class MockResponse(BaseModel):
    """Mock response for testing."""

    label: str


class MockMultiResponse(BaseModel):
    """Mock multi-label response for testing."""

    labels: list[str]


class MockClassifier:
    """Mock classifier that returns predetermined responses."""

    def __init__(self, definition, responses=None, multi_responses=None):
        self.definition = definition
        self.responses = responses or []
        self.multi_responses = multi_responses or []

    def batch_predict(self, texts, n_jobs=None):  # noqa: ARG002
        """Return mock responses for single-label classification."""
        return [MockResponse(label=label) for label in self.responses[: len(texts)]]

    def batch_predict_multi(self, texts, n_jobs=None):  # noqa: ARG002
        """Return mock responses for multi-label classification."""
        return [
            MockMultiResponse(labels=labels)
            for labels in self.multi_responses[: len(texts)]
        ]


def test_evaluate_classifier_single_label():
    """Test evaluation of single-label classification."""
    # Load definition
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)

    # Load eval set
    eval_set_path = os.path.join(current_dir, "example_evalset.yaml")
    eval_set = EvalSet.from_yaml(eval_set_path)

    # Create mock classifier with perfect predictions
    perfect_responses = [
        ex.expected_label for ex in eval_set.examples if ex.expected_label
    ]
    mock_classifier = MockClassifier(definition, responses=perfect_responses)

    # Evaluate
    result = evaluate_classifier(mock_classifier, eval_set)

    # Check results
    assert result.accuracy == 1.0
    assert result.correct_predictions == len(perfect_responses)
    assert result.total_examples == len(perfect_responses)

    # Test with display function (no assertions, just ensure it runs)
    display_evaluation_results(result)

    # Test with imperfect predictions
    # Mix up some of the predictions
    imperfect_responses = perfect_responses.copy()
    imperfect_responses[0] = "coding"  # Change first question to coding
    imperfect_responses[5] = "question"  # Change a scheduling to question

    mock_classifier = MockClassifier(definition, responses=imperfect_responses)
    result = evaluate_classifier(mock_classifier, eval_set)

    # Check results are less than perfect
    assert result.accuracy < 1.0
    assert result.correct_predictions < len(perfect_responses)


def test_evaluate_classifier_multi_label():
    """Test evaluation with multi-label classification."""
    # Create a multi-label eval set from the single-label one
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)

    eval_set_path = os.path.join(current_dir, "example_evalset.yaml")
    single_eval_set = EvalSet.from_yaml(eval_set_path)

    # Convert to multi-label (using the same labels for simplicity)
    multi_eval_set = EvalSet(
        name="Multi-label test",
        description="Test multi-label evaluation",
        classification_type="multi",
        examples=[
            {
                "text": ex.text,
                "expected_labels": [ex.expected_label] if ex.expected_label else None,
            }
            for ex in single_eval_set.examples
        ],
    )

    # Create mock classifier with perfect predictions
    perfect_responses = [
        [ex.expected_label] for ex in single_eval_set.examples if ex.expected_label
    ]
    mock_classifier = MockClassifier(definition, multi_responses=perfect_responses)

    # Evaluate
    result = evaluate_classifier(mock_classifier, multi_eval_set)

    # Check results
    assert result.accuracy == 1.0
    assert result.correct_predictions == len(perfect_responses)
    assert result.total_examples == len(perfect_responses)

    # Test with imperfect predictions
    imperfect_responses = perfect_responses.copy()
    imperfect_responses[0] = ["coding", "question"]  # Add an extra label
    imperfect_responses[5] = ["question"]  # Change a label

    mock_classifier = MockClassifier(definition, multi_responses=imperfect_responses)
    result = evaluate_classifier(mock_classifier, multi_eval_set)

    # Check results are less than perfect
    assert result.accuracy < 1.0
    assert result.correct_predictions < len(perfect_responses)


def test_eval_set_validation():
    """Test validation of eval sets against classification definition."""
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)

    eval_set_path = os.path.join(current_dir, "example_evalset.yaml")
    eval_set = EvalSet.from_yaml(eval_set_path)

    # Valid evaluation set
    assert eval_set.validate_against_definition(definition) is True

    # Invalid evaluation set with unknown label
    invalid_eval_set = EvalSet(
        name="Invalid test",
        description="Test with invalid label",
        classification_type="single",
        examples=[{"text": "Hello", "expected_label": "unknown_label"}],
    )

    assert invalid_eval_set.validate_against_definition(definition) is False
