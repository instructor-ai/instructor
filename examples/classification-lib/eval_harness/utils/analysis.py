"""
Advanced analysis utilities for classification evaluation.

Includes:
- Bootstrapped confidence intervals
- Cost and latency analysis
- Detailed confusion analysis and error pattern detection
"""

import os
import time
import random
from typing import Any, Optional
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from sklearn.metrics import confusion_matrix

# Add parent directory to path
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class BootstrapAnalyzer:
    """Performs bootstrapped confidence interval analysis on evaluation results."""

    def __init__(self, n_resamples: int = 1000, confidence_level: float = 0.95):
        """
        Initialize the bootstrap analyzer.

        Parameters
        ----------
        n_resamples : int
            Number of bootstrap resamples to generate
        confidence_level : float
            Confidence level for interval calculation (between 0 and 1)
        """
        self.n_resamples = n_resamples
        self.confidence_level = confidence_level
        self.console = Console()

    def analyze(self, eval_result) -> dict[str, Any]:
        """
        Generate bootstrapped confidence intervals for metrics in the evaluation result.

        Parameters
        ----------
        eval_result : Union[EvalResult, Dict[str, Any]]
            The evaluation result to analyze, either as EvalResult object or dictionary

        Returns
        -------
        Dict[str, Any]
            Dictionary containing metrics with confidence intervals
        """
        self.console.print(
            "[bold cyan]Generating bootstrapped confidence intervals...[/bold cyan]"
        )

        # Extract prediction data
        if hasattr(eval_result, "predictions"):
            # It's an EvalResult object
            predictions = eval_result.predictions
        elif isinstance(eval_result, dict) and "predictions" in eval_result:
            # It's a dictionary with predictions key
            predictions = eval_result["predictions"]
        else:
            # Handle case where we don't have predictions
            # Create a simplified result with just the accuracy
            accuracy = (
                eval_result.get("accuracy", 0) if isinstance(eval_result, dict) else 0
            )
            return {
                "accuracy": {
                    "mean": accuracy,
                    "lower": max(0, accuracy * 0.95),  # Simple estimate
                    "upper": min(1.0, accuracy * 1.05),  # Simple estimate
                },
                "f1_scores": {},
            }

        # Define statistic functions
        def calc_accuracy(data):
            correct = sum(1 for p in data if p["expected"] == p["predicted"])
            return correct / len(data) if data else 0

        def calc_label_f1(data, label):
            # Extract true positives, false positives, false negatives
            tp = sum(
                1 for p in data if p["expected"] == label and p["predicted"] == label
            )
            fp = sum(
                1 for p in data if p["expected"] != label and p["predicted"] == label
            )
            fn = sum(
                1 for p in data if p["expected"] == label and p["predicted"] != label
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            return f1

        # Get all labels
        all_labels = set()
        for p in predictions:
            if isinstance(p["expected"], list):
                all_labels.update(p["expected"])
            else:
                all_labels.add(p["expected"])

        # Perform bootstrap analysis
        results = {}

        # Overall accuracy
        accuracy_samples = []
        for _ in range(self.n_resamples):
            # Resample with replacement
            sample = random.choices(predictions, k=len(predictions))
            accuracy_samples.append(calc_accuracy(sample))

        # Calculate confidence interval
        lower_idx = int((1 - self.confidence_level) / 2 * self.n_resamples)
        upper_idx = int((1 - (1 - self.confidence_level) / 2) * self.n_resamples)
        sorted_acc = sorted(accuracy_samples)
        results["accuracy"] = {
            "mean": np.mean(accuracy_samples),
            "lower": sorted_acc[lower_idx],
            "upper": sorted_acc[upper_idx],
        }

        # Per-label F1 scores
        results["f1_scores"] = {}
        for label in all_labels:
            f1_samples = []
            for _ in range(self.n_resamples):
                sample = random.choices(predictions, k=len(predictions))
                f1_samples.append(calc_label_f1(sample, label))

            sorted_f1 = sorted(f1_samples)
            results["f1_scores"][label] = {
                "mean": np.mean(f1_samples),
                "lower": sorted_f1[lower_idx],
                "upper": sorted_f1[upper_idx],
            }

        return results

    def display_results(self, bootstrap_results: dict[str, Any]) -> None:
        """Display bootstrapped confidence intervals in a rich table."""
        accuracy = bootstrap_results["accuracy"]
        f1_scores = bootstrap_results["f1_scores"]

        # Accuracy table
        accuracy_table = Table(
            title=f"Bootstrapped Accuracy ({self.confidence_level * 100:.0f}% CI)"
        )
        accuracy_table.add_column("Metric", style="cyan")
        accuracy_table.add_column("Mean", style="green")
        accuracy_table.add_column("Lower Bound", style="yellow")
        accuracy_table.add_column("Upper Bound", style="yellow")

        accuracy_table.add_row(
            "Accuracy",
            f"{accuracy['mean']:.2%}",
            f"{accuracy['lower']:.2%}",
            f"{accuracy['upper']:.2%}",
        )

        self.console.print(accuracy_table)

        # F1 score table
        f1_table = Table(
            title=f"Bootstrapped F1 Scores ({self.confidence_level * 100:.0f}% CI)"
        )
        f1_table.add_column("Label", style="cyan")
        f1_table.add_column("Mean F1", style="green")
        f1_table.add_column("Lower Bound", style="yellow")
        f1_table.add_column("Upper Bound", style="yellow")

        for label, stats in f1_scores.items():
            f1_table.add_row(
                label,
                f"{stats['mean']:.2%}",
                f"{stats['lower']:.2%}",
                f"{stats['upper']:.2%}",
            )

        self.console.print(f1_table)

    def plot_bootstrap_distributions(
        self, bootstrap_results: dict[str, Any], save_path: Optional[str] = None
    ) -> None:
        """
        Plot the bootstrap distributions for accuracy and F1 scores.

        Parameters
        ----------
        bootstrap_results : Dict[str, Any]
            The bootstrap analysis results
        save_path : Optional[str]
            Path to save the plot (if None, the plot is shown)
        """
        # Check if we have the necessary data
        if "accuracy" not in bootstrap_results:
            self.console.print(
                "[yellow]Insufficient data for bootstrap distribution plot[/yellow]"
            )
            return

        # Create subplots: 1 for accuracy, 1 for each label's F1 score
        n_plots = 1 + len(bootstrap_results.get("f1_scores", {}))
        fig, axs = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))

        if n_plots == 1:
            axs = [axs]  # Convert to list for consistent indexing

        # Plot accuracy distribution - we don't have samples, so we'll just plot vertical lines
        ax = axs[0]
        accuracy = bootstrap_results["accuracy"]

        # Create a simple normal distribution around the mean for visualization purposes
        if "mean" in accuracy:
            x = np.linspace(
                max(0, accuracy["mean"] - 0.2), min(1.0, accuracy["mean"] + 0.2), 100
            )
            # Simulate a normal distribution around the mean
            width = (
                accuracy.get("upper", accuracy["mean"] * 1.05)
                - accuracy.get("lower", accuracy["mean"] * 0.95)
            ) / 4
            y = np.exp(-0.5 * ((x - accuracy["mean"]) / width) ** 2)
            ax.plot(x, y, color="blue")
            ax.fill_between(x, y, alpha=0.3)

            ax.axvline(accuracy["mean"], color="red", linestyle="-", label="Mean")

            if "lower" in accuracy and "upper" in accuracy:
                ax.axvline(
                    accuracy["lower"],
                    color="orange",
                    linestyle="--",
                    label=f"{self.confidence_level * 100:.0f}% CI",
                )
                ax.axvline(accuracy["upper"], color="orange", linestyle="--")

            ax.set_title("Bootstrapped Accuracy Estimate")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Density")
            ax.legend()

        # Plot F1 score distributions
        for i, (label, stats) in enumerate(
            bootstrap_results.get("f1_scores", {}).items(), 1
        ):
            if i >= len(axs):
                break

            ax = axs[i]

            # Create a simple normal distribution around the mean for visualization purposes
            if "mean" in stats:
                x = np.linspace(
                    max(0, stats["mean"] - 0.2), min(1.0, stats["mean"] + 0.2), 100
                )
                # Simulate a normal distribution
                width = (
                    stats.get("upper", stats["mean"] * 1.05)
                    - stats.get("lower", stats["mean"] * 0.95)
                ) / 4
                y = np.exp(-0.5 * ((x - stats["mean"]) / width) ** 2)
                ax.plot(x, y, color="blue")
                ax.fill_between(x, y, alpha=0.3)

                ax.axvline(stats["mean"], color="red", linestyle="-", label="Mean")

                if "lower" in stats and "upper" in stats:
                    ax.axvline(
                        stats["lower"],
                        color="orange",
                        linestyle="--",
                        label=f"{self.confidence_level * 100:.0f}% CI",
                    )
                    ax.axvline(stats["upper"], color="orange", linestyle="--")

                ax.set_title(f"Bootstrapped F1 Score Estimate: {label}")
                ax.set_xlabel("F1 Score")
                ax.set_ylabel("Density")
                ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class CostLatencyAnalyzer:
    """Analyzes cost and latency metrics for classification operations."""

    # Cost per 1K tokens for different models (in USD)
    MODEL_COSTS = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4o-mini": {"input": 0.0015, "output": 0.0060},
        "gpt-4o": {"input": 0.0050, "output": 0.0150},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    }

    def __init__(self):
        """Initialize the cost and latency analyzer."""
        self.console = Console()
        self.token_counts = defaultdict(lambda: {"input": 0, "output": 0})
        self.latencies = defaultdict(list)
        self.timing_data = defaultdict(list)

    def start_timing(self, model: str, text: str) -> int:
        """
        Start timing a classification request.

        Parameters
        ----------
        model : str
            The model name
        text : str
            The text being classified

        Returns
        -------
        int
            Timestamp for tracking
        """
        timestamp = int(time.time() * 1000)
        self.timing_data[timestamp] = {
            "model": model,
            "text": text,
            "start_time": timestamp,
            "end_time": None,
            "input_tokens": None,
            "output_tokens": None,
        }
        return timestamp

    def end_timing(self, timestamp: int, input_tokens: int, output_tokens: int) -> None:
        """
        End timing for a classification request and record token usage.

        Parameters
        ----------
        timestamp : int
            The timestamp returned by start_timing
        input_tokens : int
            Number of input tokens used
        output_tokens : int
            Number of output tokens used
        """
        end_time = int(time.time() * 1000)

        if timestamp in self.timing_data:
            data = self.timing_data[timestamp]
            data["end_time"] = end_time
            data["input_tokens"] = input_tokens
            data["output_tokens"] = output_tokens

            model = data["model"]
            latency = end_time - data["start_time"]

            # Update token counts
            self.token_counts[model]["input"] += input_tokens
            self.token_counts[model]["output"] += output_tokens

            # Update latencies
            self.latencies[model].append(latency)

    def calculate_costs(self) -> dict[str, dict[str, float]]:
        """
        Calculate costs for each model based on recorded token usage.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of cost metrics by model
        """
        costs = {}

        for model, tokens in self.token_counts.items():
            if model in self.MODEL_COSTS:
                input_cost = tokens["input"] / 1000 * self.MODEL_COSTS[model]["input"]
                output_cost = (
                    tokens["output"] / 1000 * self.MODEL_COSTS[model]["output"]
                )
                total_cost = input_cost + output_cost

                costs[model] = {
                    "input_tokens": tokens["input"],
                    "output_tokens": tokens["output"],
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost,
                }
            else:
                # Use a fallback for unknown models
                costs[model] = {
                    "input_tokens": tokens["input"],
                    "output_tokens": tokens["output"],
                    "input_cost": None,
                    "output_cost": None,
                    "total_cost": None,
                }

        return costs

    def calculate_latency_stats(self) -> dict[str, dict[str, float]]:
        """
        Calculate latency statistics for each model.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of latency metrics by model
        """
        latency_stats = {}

        for model, latencies in self.latencies.items():
            if latencies:
                latency_stats[model] = {
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": sum(latencies) / len(latencies),
                    "median": sorted(latencies)[len(latencies) // 2],
                    "p90": sorted(latencies)[int(len(latencies) * 0.9)],
                    "p95": sorted(latencies)[int(len(latencies) * 0.95)],
                    "p99": sorted(latencies)[int(len(latencies) * 0.99)]
                    if len(latencies) >= 100
                    else None,
                    "count": len(latencies),
                }
            else:
                latency_stats[model] = {}

        return latency_stats

    def display_cost_analysis(self, costs: dict[str, dict[str, float]]) -> None:
        """Display cost analysis in a rich table."""
        cost_table = Table(title="Cost Analysis by Model")
        cost_table.add_column("Model", style="cyan")
        cost_table.add_column("Input Tokens", style="green")
        cost_table.add_column("Output Tokens", style="green")
        cost_table.add_column("Input Cost", style="yellow")
        cost_table.add_column("Output Cost", style="yellow")
        cost_table.add_column("Total Cost", style="bold red")

        for model, metrics in costs.items():
            input_cost = (
                f"${metrics['input_cost']:.4f}"
                if metrics["input_cost"] is not None
                else "Unknown"
            )
            output_cost = (
                f"${metrics['output_cost']:.4f}"
                if metrics["output_cost"] is not None
                else "Unknown"
            )
            total_cost = (
                f"${metrics['total_cost']:.4f}"
                if metrics["total_cost"] is not None
                else "Unknown"
            )

            cost_table.add_row(
                model,
                f"{metrics['input_tokens']:,}",
                f"{metrics['output_tokens']:,}",
                input_cost,
                output_cost,
                total_cost,
            )

        self.console.print(cost_table)

    def display_latency_analysis(
        self, latency_stats: dict[str, dict[str, float]]
    ) -> None:
        """Display latency analysis in a rich table."""
        latency_table = Table(title="Latency Analysis by Model (milliseconds)")
        latency_table.add_column("Model", style="cyan")
        latency_table.add_column("Count", style="cyan")
        latency_table.add_column("Min", style="green")
        latency_table.add_column("Mean", style="yellow")
        latency_table.add_column("Median", style="yellow")
        latency_table.add_column("P90", style="red")
        latency_table.add_column("P95", style="red")
        latency_table.add_column("Max", style="bold red")

        for model, stats in latency_stats.items():
            if stats:
                latency_table.add_row(
                    model,
                    f"{stats['count']}",
                    f"{stats['min']:.0f}",
                    f"{stats['mean']:.0f}",
                    f"{stats['median']:.0f}",
                    f"{stats['p90']:.0f}",
                    f"{stats['p95']:.0f}",
                    f"{stats['max']:.0f}",
                )

        self.console.print(latency_table)

    def plot_latency_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot latency distributions for each model.

        Parameters
        ----------
        save_path : Optional[str]
            Path to save the plot (if None, the plot is shown)
        """
        plt.figure(figsize=(12, 6))

        # Create violin plots for each model
        data = []
        labels = []

        for model, latencies in self.latencies.items():
            if latencies:
                data.append(latencies)
                labels.append(model)

        if data:
            plt.violinplot(data, showmedians=True)
            plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
            plt.ylabel("Latency (ms)")
            plt.title("Latency Distribution by Model")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()

    def plot_cost_efficiency(
        self,
        costs: dict[str, dict[str, float]],
        accuracy_by_model: dict[str, float],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot cost efficiency (accuracy per dollar) for each model.

        Parameters
        ----------
        costs : Dict[str, Dict[str, float]]
            Cost analysis results
        accuracy_by_model : Dict[str, float]
            Accuracy for each model
        save_path : Optional[str]
            Path to save the plot (if None, the plot is shown)
        """
        models = []
        total_costs = []
        accuracies = []
        efficiency = []

        for model, cost_data in costs.items():
            if model in accuracy_by_model and cost_data["total_cost"] is not None:
                models.append(model)
                total_cost = cost_data["total_cost"]
                accuracy = accuracy_by_model[model]

                total_costs.append(total_cost)
                accuracies.append(accuracy)

                # Calculate efficiency: accuracy percentage per dollar
                eff = accuracy / total_cost if total_cost > 0 else 0
                efficiency.append(eff)

        if not models:
            return

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Bar chart for costs
        x = range(len(models))
        ax1.bar(x, total_costs, alpha=0.6, color="blue", label="Total Cost ($)")
        ax1.set_xlabel("Models")
        ax1.set_ylabel("Cost ($)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Second y-axis for accuracy
        ax2 = ax1.twinx()
        ax2.plot(x, accuracies, "r-", marker="o", label="Accuracy")
        ax2.set_ylabel("Accuracy", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Set x-ticks to model names
        plt.xticks(x, models, rotation=45, ha="right")

        # Add a third set of labels for efficiency
        for i, (eff, cost) in enumerate(zip(efficiency, total_costs)):
            plt.text(
                i,
                cost + 0.01,
                f"Eff: {eff:.1f}%/$",
                ha="center",
                va="bottom",
                color="green",
                fontweight="bold",
            )

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title("Cost, Accuracy, and Efficiency by Model")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class ConfusionAnalyzer:
    """Advanced confusion matrix analysis and error pattern detection."""

    def __init__(self):
        """Initialize the confusion analyzer."""
        self.console = Console()

    def analyze(self, eval_result) -> dict[str, Any]:
        """
        Perform detailed confusion analysis on evaluation results.

        Parameters
        ----------
        eval_result : Union[EvalResult, Dict[str, Any]]
            The evaluation result to analyze, either as EvalResult object or dictionary

        Returns
        -------
        Dict[str, Any]
            Dictionary containing confusion analysis results
        """
        self.console.print(
            "[bold cyan]Performing detailed confusion analysis...[/bold cyan]"
        )

        # Extract prediction data
        if hasattr(eval_result, "predictions"):
            # It's an EvalResult object
            predictions = eval_result.predictions
        elif isinstance(eval_result, dict) and "predictions" in eval_result:
            # It's a dictionary with predictions key
            predictions = eval_result["predictions"]
        else:
            # If we don't have predictions, return empty result
            self.console.print(
                "[yellow]No predictions found for confusion analysis[/yellow]"
            )
            return {}

        # Skip analysis if we have multi-label predictions
        if any(isinstance(p.get("expected"), list) for p in predictions):
            self.console.print(
                "[yellow]Confusion analysis is only available for single-label classifications[/yellow]"
            )
            return {}

        # Get all unique labels
        all_labels = set()
        for p in predictions:
            all_labels.add(p["expected"])
            all_labels.add(p["predicted"])
        all_labels = sorted(all_labels)

        # Create label index mapping
        label_to_idx = {label: i for i, label in enumerate(all_labels)}

        # Prepare data for confusion matrix
        y_true = [label_to_idx[p["expected"]] for p in predictions]
        y_pred = [label_to_idx[p["predicted"]] for p in predictions]

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate normalized confusion matrix (by row)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0

        # Identify most confused pairs
        confused_pairs = []
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append(
                        {
                            "true_label": all_labels[i],
                            "predicted_label": all_labels[j],
                            "count": int(cm[i, j]),
                            "percentage": float(cm_normalized[i, j]),
                        }
                    )

        # Sort by count
        confused_pairs.sort(key=lambda x: x["count"], reverse=True)

        # Find examples for each confused pair
        confusion_examples = {}
        for pair in confused_pairs:
            true_label = pair["true_label"]
            pred_label = pair["predicted_label"]
            key = f"{true_label}→{pred_label}"

            # Find examples with this confusion
            examples = [
                p["text"]
                for p in predictions
                if p["expected"] == true_label and p["predicted"] == pred_label
            ]

            confusion_examples[key] = examples

        # Group predictions by expected label
        predictions_by_label = defaultdict(list)
        for p in predictions:
            predictions_by_label[p["expected"]].append(p)

        # Analyze error patterns: look for common words/patterns in incorrect predictions
        error_patterns = {}
        for label, preds in predictions_by_label.items():
            correct = [p["text"] for p in preds if p["predicted"] == p["expected"]]
            incorrect = [p["text"] for p in preds if p["predicted"] != p["expected"]]

            # Skip if no incorrect predictions
            if not incorrect:
                continue

            # Simple pattern analysis - count word frequencies
            # In a full implementation, we would use more sophisticated NLP techniques
            error_words = Counter()
            for text in incorrect:
                words = text.lower().split()
                error_words.update(words)

            correct_words = Counter()
            for text in correct:
                words = text.lower().split()
                correct_words.update(words)

            # Find words more common in errors (crude approximation)
            distinctive_error_words = {}
            for word, count in error_words.items():
                if count > 1:  # Only consider words that appear multiple times
                    error_freq = count / len(incorrect)
                    correct_freq = correct_words.get(word, 0) / max(len(correct), 1)

                    if error_freq > correct_freq:
                        distinctive_error_words[word] = error_freq / max(
                            correct_freq, 0.1
                        )

            # Sort by distinctiveness ratio
            sorted_words = sorted(
                distinctive_error_words.items(), key=lambda x: x[1], reverse=True
            )
            error_patterns[label] = sorted_words[:10]  # Top 10 distinctive words

        return {
            "confusion_matrix": cm.tolist(),
            "labels": all_labels,
            "confused_pairs": confused_pairs,
            "confusion_examples": confusion_examples,
            "error_patterns": error_patterns,
        }

    def display_results(self, confusion_analysis: dict[str, Any]) -> None:
        """Display confusion analysis results using rich tables and panels."""
        if not confusion_analysis:
            return

        labels = confusion_analysis["labels"]
        cm = confusion_analysis["confusion_matrix"]
        confused_pairs = confusion_analysis["confused_pairs"]
        confusion_examples = confusion_analysis["confusion_examples"]
        error_patterns = confusion_analysis["error_patterns"]

        # Display most confused pairs
        if confused_pairs:
            confused_table = Table(title="Most Confused Label Pairs")
            confused_table.add_column("True Label", style="cyan")
            confused_table.add_column("Predicted Label", style="yellow")
            confused_table.add_column("Count", style="green")
            confused_table.add_column("Percentage", style="red")

            for pair in confused_pairs[:10]:  # Show top 10
                confused_table.add_row(
                    pair["true_label"],
                    pair["predicted_label"],
                    str(pair["count"]),
                    f"{pair['percentage']:.1%}",
                )

            self.console.print(confused_table)

        # Display examples of confused pairs
        if confusion_examples and confused_pairs:
            self.console.print("\n[bold]Examples of Confused Pairs:[/bold]")

            for pair in confused_pairs[:5]:  # Show top 5
                true_label = pair["true_label"]
                pred_label = pair["predicted_label"]
                key = f"{true_label}→{pred_label}"

                if key in confusion_examples and confusion_examples[key]:
                    examples = confusion_examples[key][:3]  # Show up to 3 examples

                    panel_title = f"[bold]{true_label} mistaken as {pred_label} ({pair['count']} times)[/bold]"
                    panel_content = "\n\n".join(
                        [
                            f"• {ex[:100]}..." if len(ex) > 100 else f"• {ex}"
                            for ex in examples
                        ]
                    )

                    self.console.print(
                        Panel(panel_content, title=panel_title, border_style="yellow")
                    )

        # Display error patterns
        if error_patterns:
            self.console.print("\n[bold]Possible Error Patterns:[/bold]")

            for label, patterns in error_patterns.items():
                if patterns:
                    words = [f"{word} ({ratio:.1f}x)" for word, ratio in patterns[:5]]
                    pattern_text = Text(f"{label}: {', '.join(words)}")
                    self.console.print(pattern_text)

    def plot_confusion_matrix(
        self, confusion_analysis: dict[str, Any], save_path: Optional[str] = None
    ) -> None:
        """
        Plot a heatmap of the confusion matrix.

        Parameters
        ----------
        confusion_analysis : Dict[str, Any]
            The confusion analysis results
        save_path : Optional[str]
            Path to save the plot (if None, the plot is shown)
        """
        if not confusion_analysis:
            return

        labels = confusion_analysis["labels"]
        cm = np.array(confusion_analysis["confusion_matrix"])

        # Create normalized confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0

        plt.figure(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(
            cm_normalized,
            annot=cm,  # Show raw counts in cells
            fmt="d",  # Integer format for counts
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_error_distribution(
        self, confusion_analysis: dict[str, Any], save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of errors by label.

        Parameters
        ----------
        confusion_analysis : Dict[str, Any]
            The confusion analysis results
        save_path : Optional[str]
            Path to save the plot (if None, the plot is shown)
        """
        if not confusion_analysis:
            return

        labels = confusion_analysis["labels"]
        cm = np.array(confusion_analysis["confusion_matrix"])

        # Calculate errors for each true label
        errors_by_label = []
        for i, label in enumerate(labels):
            total = cm[i].sum()
            correct = cm[i, i]
            error_rate = 1 - (correct / total) if total > 0 else 0
            errors_by_label.append((label, error_rate, total - correct))

        # Sort by error rate
        errors_by_label.sort(key=lambda x: x[1], reverse=True)

        # Prepare data for plotting
        plot_labels = [x[0] for x in errors_by_label]
        error_rates = [x[1] for x in errors_by_label]
        error_counts = [x[2] for x in errors_by_label]

        plt.figure(figsize=(12, 6))

        # Create bar chart
        bars = plt.bar(plot_labels, error_rates, color="red", alpha=0.7)

        # Add count labels on top of bars
        for bar, count in zip(bars, error_counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{count}",
                ha="center",
                va="bottom",
            )

        plt.ylabel("Error Rate")
        plt.title("Error Distribution by True Label")
        plt.ylim(0, max(error_rates) * 1.2)  # Add some space for labels
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
