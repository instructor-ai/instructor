from __future__ import annotations

from collections import defaultdict

from rich.console import Console
from rich.table import Table

from classify import Classifier, AsyncClassifier
from schema import EvalResult, EvalSet


def _calculate_metrics(
    true_positives: int, false_positives: int, false_negatives: int
) -> dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _build_confusion_matrix(
    predictions: list[dict[str, str]], all_labels: set[str]
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix from predictions."""
    matrix = {
        actual: {predicted: 0 for predicted in all_labels} for actual in all_labels
    }

    for prediction in predictions:
        actual = prediction["expected"]
        predicted = prediction["predicted"]

        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1

    return matrix


def evaluate_classifier(
    classifier: Classifier, eval_set: EvalSet, n_jobs: int | None = None
) -> EvalResult:
    """Evaluate a classifier on a set of examples and return metrics."""
    if eval_set.classification_type == "single":
        # Filter for examples with expected_label
        valid_examples = [ex for ex in eval_set.examples if ex.expected_label]

        # Run batch prediction
        if len(valid_examples) > 0:
            predictions = classifier.batch_predict(
                [ex.text for ex in valid_examples], n_jobs=n_jobs
            )
        else:
            return EvalResult(
                eval_set_name=eval_set.name,
                total_examples=0,
                correct_predictions=0,
                accuracy=0.0,
                per_label_metrics={},
                confusion_matrix={},
                predictions=[],
            )

        # Process results
        all_labels = {ld.label for ld in classifier.definition.label_definitions}

        prediction_details = []
        correct_count = 0
        label_counts = defaultdict(int)  # For total occurrences of each label
        tp_counts = defaultdict(int)  # True positives by label
        fp_counts = defaultdict(int)  # False positives by label
        fn_counts = defaultdict(int)  # False negatives by label

        for _i, (example, pred) in enumerate(zip(valid_examples, predictions)):
            expected = example.expected_label
            predicted = pred.label

            prediction_details.append(
                {
                    "text": example.text,
                    "expected": expected,
                    "predicted": predicted,
                }
            )

            # Update counts
            label_counts[expected] += 1

            if predicted == expected:
                correct_count += 1
                tp_counts[predicted] += 1
            else:
                fp_counts[predicted] += 1
                fn_counts[expected] += 1

        # Calculate per-label metrics
        per_label_metrics = {}
        for label in all_labels:
            per_label_metrics[label] = _calculate_metrics(
                tp_counts[label], fp_counts[label], fn_counts[label]
            )

        # Build confusion matrix
        confusion_matrix = _build_confusion_matrix(prediction_details, all_labels)

        # Calculate overall accuracy
        accuracy = correct_count / len(valid_examples) if valid_examples else 0

        return EvalResult(
            eval_set_name=eval_set.name,
            total_examples=len(valid_examples),
            correct_predictions=correct_count,
            accuracy=accuracy,
            per_label_metrics=per_label_metrics,
            confusion_matrix=confusion_matrix,
            predictions=prediction_details,
        )

    else:  # Multi-label classification
        # Filter for examples with expected_labels
        valid_examples = [ex for ex in eval_set.examples if ex.expected_labels]

        # Run batch prediction
        if len(valid_examples) > 0:
            predictions = classifier.batch_predict_multi(
                [ex.text for ex in valid_examples], n_jobs=n_jobs
            )
        else:
            return EvalResult(
                eval_set_name=eval_set.name,
                total_examples=0,
                correct_predictions=0,
                accuracy=0.0,
                per_label_metrics={},
                confusion_matrix={},
                predictions=[],
            )

        # Multi-label evaluation is more complex
        # We'll consider a prediction correct if all labels match exactly

        prediction_details = []
        correct_count = 0

        # For multi-label, we don't calculate a traditional confusion matrix
        # Instead, we track per-label metrics
        all_labels = {ld.label for ld in classifier.definition.label_definitions}

        label_tp = defaultdict(int)  # True positives by label
        label_fp = defaultdict(int)  # False positives by label
        label_fn = defaultdict(int)  # False negatives by label

        for example, pred in zip(valid_examples, predictions):
            expected_set = set(example.expected_labels)
            predicted_set = set(pred.labels)

            prediction_details.append(
                {
                    "text": example.text,
                    "expected": example.expected_labels,
                    "predicted": pred.labels,
                }
            )

            # Update counts for each label
            for label in all_labels:
                if label in expected_set and label in predicted_set:
                    label_tp[label] += 1
                elif label not in expected_set and label in predicted_set:
                    label_fp[label] += 1
                elif label in expected_set and label not in predicted_set:
                    label_fn[label] += 1

            # A prediction is correct if all labels match exactly
            if expected_set == predicted_set:
                correct_count += 1

        # Calculate per-label metrics
        per_label_metrics = {}
        for label in all_labels:
            per_label_metrics[label] = _calculate_metrics(
                label_tp[label], label_fp[label], label_fn[label]
            )

        # For multi-label, we don't have a traditional confusion matrix
        # We'll provide a placeholder - custom visualization would be better
        confusion_matrix = {}

        # Calculate overall accuracy (exact matches)
        accuracy = correct_count / len(valid_examples) if valid_examples else 0

        return EvalResult(
            eval_set_name=eval_set.name,
            total_examples=len(valid_examples),
            correct_predictions=correct_count,
            accuracy=accuracy,
            per_label_metrics=per_label_metrics,
            confusion_matrix=confusion_matrix,
            predictions=prediction_details,
        )


async def evaluate_classifier_async(
    classifier: AsyncClassifier, eval_set: EvalSet, n_jobs: int | None = 5
) -> EvalResult:
    """Evaluate an async classifier on a set of examples and return metrics."""
    if eval_set.classification_type == "single":
        # Filter for examples with expected_label
        valid_examples = [ex for ex in eval_set.examples if ex.expected_label]

        # Run batch prediction
        if len(valid_examples) > 0:
            predictions = await classifier.batch_predict(
                [ex.text for ex in valid_examples], n_jobs=n_jobs
            )
        else:
            return EvalResult(
                eval_set_name=eval_set.name,
                total_examples=0,
                correct_predictions=0,
                accuracy=0.0,
                per_label_metrics={},
                confusion_matrix={},
                predictions=[],
            )

        # Process results
        all_labels = {ld.label for ld in classifier.definition.label_definitions}

        prediction_details = []
        correct_count = 0
        label_counts = defaultdict(int)  # For total occurrences of each label
        tp_counts = defaultdict(int)  # True positives by label
        fp_counts = defaultdict(int)  # False positives by label
        fn_counts = defaultdict(int)  # False negatives by label

        for _i, (example, pred) in enumerate(zip(valid_examples, predictions)):
            expected = example.expected_label
            predicted = pred.label

            prediction_details.append(
                {
                    "text": example.text,
                    "expected": expected,
                    "predicted": predicted,
                }
            )

            # Update counts
            label_counts[expected] += 1

            if predicted == expected:
                correct_count += 1
                tp_counts[predicted] += 1
            else:
                fp_counts[predicted] += 1
                fn_counts[expected] += 1

        # Calculate per-label metrics
        per_label_metrics = {}
        for label in all_labels:
            per_label_metrics[label] = _calculate_metrics(
                tp_counts[label], fp_counts[label], fn_counts[label]
            )

        # Build confusion matrix
        confusion_matrix = _build_confusion_matrix(prediction_details, all_labels)

        # Calculate overall accuracy
        accuracy = correct_count / len(valid_examples) if valid_examples else 0

        return EvalResult(
            eval_set_name=eval_set.name,
            total_examples=len(valid_examples),
            correct_predictions=correct_count,
            accuracy=accuracy,
            per_label_metrics=per_label_metrics,
            confusion_matrix=confusion_matrix,
            predictions=prediction_details,
        )

    else:  # Multi-label
        # Filter for examples with expected_labels
        valid_examples = [ex for ex in eval_set.examples if ex.expected_labels]

        # Run batch prediction
        if len(valid_examples) > 0:
            predictions = await classifier.batch_predict_multi(
                [ex.text for ex in valid_examples], n_jobs=n_jobs
            )
        else:
            return EvalResult(
                eval_set_name=eval_set.name,
                total_examples=0,
                correct_predictions=0,
                accuracy=0.0,
                per_label_metrics={},
                confusion_matrix={},
                predictions=[],
            )

        # Multi-label evaluation logic (similar to sync version)
        prediction_details = []
        correct_count = 0

        all_labels = {ld.label for ld in classifier.definition.label_definitions}

        label_tp = defaultdict(int)
        label_fp = defaultdict(int)
        label_fn = defaultdict(int)

        for example, pred in zip(valid_examples, predictions):
            expected_set = set(example.expected_labels)
            predicted_set = set(pred.labels)

            prediction_details.append(
                {
                    "text": example.text,
                    "expected": example.expected_labels,
                    "predicted": pred.labels,
                }
            )

            # Update counts for each label
            for label in all_labels:
                if label in expected_set and label in predicted_set:
                    label_tp[label] += 1
                elif label not in expected_set and label in predicted_set:
                    label_fp[label] += 1
                elif label in expected_set and label not in predicted_set:
                    label_fn[label] += 1

            # A prediction is correct if all labels match exactly
            if expected_set == predicted_set:
                correct_count += 1

        # Calculate per-label metrics
        per_label_metrics = {}
        for label in all_labels:
            per_label_metrics[label] = _calculate_metrics(
                label_tp[label], label_fp[label], label_fn[label]
            )

        # For multi-label, we don't use a traditional confusion matrix
        confusion_matrix = {}

        # Calculate overall accuracy (exact matches)
        accuracy = correct_count / len(valid_examples) if valid_examples else 0

        return EvalResult(
            eval_set_name=eval_set.name,
            total_examples=len(valid_examples),
            correct_predictions=correct_count,
            accuracy=accuracy,
            per_label_metrics=per_label_metrics,
            confusion_matrix=confusion_matrix,
            predictions=prediction_details,
        )


def display_evaluation_results(result: EvalResult, detailed: bool = False) -> None:
    """Display evaluation results using rich tables."""
    console = Console()

    # Main results table
    main_table = Table(title=f"Evaluation Results: {result.eval_set_name}")
    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Value", style="green")

    main_table.add_row("Total Examples", str(result.total_examples))
    main_table.add_row("Correct Predictions", str(result.correct_predictions))
    main_table.add_row(
        "Accuracy", f"{result.accuracy:.2%}" if result.total_examples > 0 else "N/A"
    )

    console.print(main_table)

    # Per-label metrics table
    if result.per_label_metrics:
        metrics_table = Table(title="Per-Label Metrics")
        metrics_table.add_column("Label", style="cyan")
        metrics_table.add_column("Precision", style="green")
        metrics_table.add_column("Recall", style="green")
        metrics_table.add_column("F1 Score", style="green")

        for label, metrics in result.per_label_metrics.items():
            metrics_table.add_row(
                label,
                f"{metrics['precision']:.2%}",
                f"{metrics['recall']:.2%}",
                f"{metrics['f1']:.2%}",
            )

        console.print(metrics_table)

    # Confusion matrix (for single-label only)
    if result.confusion_matrix:
        console.print("\n[bold]Confusion Matrix:[/bold]")

        matrix_table = Table(show_header=True)
        matrix_table.add_column("Actual ↓ / Predicted →", style="cyan")

        # Add columns for each label
        labels = list(next(iter(result.confusion_matrix.values())).keys())
        for label in labels:
            matrix_table.add_column(label, style="green")

        # Add rows
        for actual_label, predictions in result.confusion_matrix.items():
            row = [actual_label]
            for pred_label in labels:
                count = predictions.get(pred_label, 0)
                # Highlight diagonal (correct predictions)
                cell_style = "[bold green]" if actual_label == pred_label else ""
                row.append(f"{cell_style}{count}")

            matrix_table.add_row(*row)

        console.print(matrix_table)

    # Detailed prediction results
    if detailed and result.predictions:
        console.print("\n[bold]Detailed Predictions:[/bold]")

        detail_table = Table(title="Predictions")
        detail_table.add_column("Text", style="cyan", no_wrap=False, width=40)
        detail_table.add_column("Expected", style="green")
        detail_table.add_column("Predicted", style="yellow")
        detail_table.add_column("Correct", style="bold")

        for pred in result.predictions:
            text = pred["text"]
            expected = pred["expected"]
            predicted = pred["predicted"]

            # Handle both single and multi-label cases
            if isinstance(expected, list) and isinstance(predicted, list):
                expected_str = ", ".join(expected)
                predicted_str = ", ".join(predicted)
                correct = set(expected) == set(predicted)
            else:
                expected_str = str(expected)
                predicted_str = str(predicted)
                correct = expected == predicted

            # Truncate long text
            if len(text) > 60:
                text = text[:57] + "..."

            detail_table.add_row(
                text,
                expected_str,
                predicted_str,
                "✓" if correct else "✗",
            )

        console.print(detail_table)
