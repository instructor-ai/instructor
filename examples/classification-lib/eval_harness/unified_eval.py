#!/usr/bin/env python3
"""
Unified evaluation framework for classification models.

This script provides a comprehensive framework for:
1. Running evaluations on multiple models and datasets
2. Tracking and analyzing cost and latency metrics
3. Generating statistical confidence intervals
4. Analyzing error patterns and confusion matrices
5. Visualizing results and comparisons

Usage:
  python unified_eval.py --config configs/evaluation.yaml
"""

import os
import sys
import argparse
import yaml
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import instructor
from openai import OpenAI, AsyncOpenAI
from schema import ClassificationDefinition, EvalSet, EvalResult

# Import local utilities
from utils.analysis import (
    BootstrapAnalyzer, 
    CostLatencyAnalyzer, 
    ConfusionAnalyzer
)


class UnifiedEvaluator:
    """
    Unified evaluation framework for classification models.
    
    This class handles:
    - Loading models, definitions, and evaluation sets
    - Running synchronous or asynchronous evaluations
    - Collecting metrics on accuracy, cost, and latency
    - Generating comprehensive analysis and visualizations
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the unified evaluator with a configuration file.
        
        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file
        """
        self.console = Console()
        
        # Load configuration
        self.console.print(f"[green]Loading configuration from {config_path}[/green]")
        self.config = self._load_config(config_path)
        
        # Set up paths
        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        self.output_dir = os.path.join(
            self.base_dir, 
            self.config.get("output_dir", "results")
        )
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analysis"), exist_ok=True)
        
        # Set up analyzers
        self.cost_analyzer = CostLatencyAnalyzer()
        self.bootstrap_analyzer = BootstrapAnalyzer(
            n_resamples=self.config.get("bootstrap_samples", 1000),
            confidence_level=self.config.get("confidence_level", 0.95)
        )
        self.confusion_analyzer = ConfusionAnalyzer()
        
        # Generate timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Tracking variables
        self.results_by_model = {}
        self.accuracy_by_model = {}
        self.evaluation_sets = []
        self.models = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.console.print(f"[bold red]Error loading configuration:[/bold red] {str(e)}")
            sys.exit(1)
    
    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the base directory if not absolute."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)
    
    def prepare(self) -> None:
        """
        Prepare for evaluation by loading all necessary components.
        
        This includes:
        - Loading the classification definition
        - Loading all evaluation sets
        - Setting up model configurations
        """
        # Load classification definition
        definition_path = self._resolve_path(self.config["definition_path"])
        self.console.print(f"[green]Loading classification definition from {definition_path}[/green]")
        
        try:
            self.definition = ClassificationDefinition.from_yaml(definition_path)
        except Exception as e:
            self.console.print(f"[bold red]Error loading classification definition:[/bold red] {str(e)}")
            sys.exit(1)
        
        # Load evaluation sets
        self.console.print("[green]Loading evaluation sets...[/green]")
        
        for eval_set_path in self.config["eval_sets"]:
            path = self._resolve_path(eval_set_path)
            try:
                eval_set = EvalSet.from_yaml(path)
                self.evaluation_sets.append(eval_set)
                self.console.print(f"  - Loaded: {eval_set.name}")
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to load {path}: {str(e)}[/yellow]")
        
        if not self.evaluation_sets:
            self.console.print("[bold red]Error: No valid evaluation sets loaded[/bold red]")
            sys.exit(1)
        
        # Set up models
        self.models = self.config["models"]
        if not self.models:
            self.console.print("[bold red]Error: No models specified in configuration[/bold red]")
            sys.exit(1)
        
        self.console.print(f"[green]Prepared {len(self.models)} models and {len(self.evaluation_sets)} evaluation sets[/green]")
    
    def run(self) -> None:
        """
        Run the evaluation process for all models and evaluation sets.
        
        This method handles:
        - Creating instrumented clients for each model
        - Running evaluations on all datasets
        - Collecting and analyzing results
        - Generating reports and visualizations
        """
        # Display evaluation plan
        self.console.print(Panel(
            f"[bold]Evaluation Plan[/bold]\n\n"
            f"Models: {', '.join(self.models)}\n"
            f"Evaluation Sets: {', '.join(es.name for es in self.evaluation_sets)}\n"
            f"Analysis: Cost & Latency, Confidence Intervals, Error Patterns\n"
            f"Output Directory: {self.output_dir}",
            title="[bold cyan]Classification Evaluation[/bold cyan]",
            border_style="cyan"
        ))
        
        # Run instrumented evaluation for each model
        for model_name in self.models:
            self.console.print(f"\n[bold cyan]Evaluating model:[/bold cyan] {model_name}")
            
            # Initialize model-specific results
            self.results_by_model[model_name] = {}
            
            # Initialize OpenAI client with Instructor
            client = instructor.from_openai(OpenAI())
            
            # Create appropriate classifier
            from classify import Classifier
            class InstrumentedClassifier(Classifier):
                def __init__(self, definition, analyzer, model_name):
                    super().__init__(definition)
                    self.analyzer = analyzer
                    self.model_name = model_name
                
                def predict(self, text):
                    self._validate_client()
                    timestamp = self.analyzer.start_timing(self.model_name, text)
                    messages, context = self._build_messages(text)
                    result = self.client.chat.completions.create_with_completion(
                        model=self.model_name,
                        response_model=self._classification_model,
                        messages=messages,
                        context=context,
                    )
                    input_tokens = result[1].usage.prompt_tokens
                    output_tokens = result[1].usage.completion_tokens
                    self.analyzer.end_timing(timestamp, input_tokens, output_tokens)
                    return result[0]
                
                def predict_multi(self, text):
                    self._validate_client()
                    timestamp = self.analyzer.start_timing(self.model_name, text)
                    messages, context = self._build_messages(text)
                    result = self.client.chat.completions.create_with_completion(
                        model=self.model_name,
                        response_model=self._multi_classification_model,
                        messages=messages,
                        context=context,
                    )
                    input_tokens = result[1].usage.prompt_tokens
                    output_tokens = result[1].usage.completion_tokens
                    self.analyzer.end_timing(timestamp, input_tokens, output_tokens)
                    return result[0]
            
            classifier = InstrumentedClassifier(
                self.definition, 
                self.cost_analyzer,
                model_name
            )
            classifier.with_client(client)
            
            # Evaluate on each evaluation set
            for eval_set in self.evaluation_sets:
                self.console.print(f"  - Evaluating on: {eval_set.name}")
                
                # Filter for examples with expected_label/expected_labels
                if eval_set.classification_type == "single":
                    valid_examples = [ex for ex in eval_set.examples if ex.expected_label]
                    
                    # Run predictions with tracking
                    predictions = []
                    correct_count = 0
                    
                    with Progress() as progress:
                        task = progress.add_task(
                            f"Classifying with {model_name}", 
                            total=len(valid_examples)
                        )
                        
                        for example in valid_examples:
                            prediction = classifier.predict(example.text)
                            predictions.append(prediction)
                            if prediction.label == example.expected_label:
                                correct_count += 1
                            progress.update(task, advance=1)
                    
                    # Calculate accuracy
                    accuracy = correct_count / len(valid_examples) if valid_examples else 0
                    
                    # Build prediction details
                    prediction_details = []
                    for example, pred in zip(valid_examples, predictions):
                        prediction_details.append({
                            "text": example.text,
                            "expected": example.expected_label,
                            "predicted": pred.label,
                        })
                    
                    # Build result
                    result = {
                        "model": model_name,
                        "eval_set_name": eval_set.name,
                        "total_examples": len(valid_examples),
                        "correct_predictions": correct_count,
                        "accuracy": accuracy,
                        "predictions": prediction_details,
                    }
                    
                else:  # Multi-label classification
                    valid_examples = [ex for ex in eval_set.examples if ex.expected_labels]
                    
                    # Run predictions with tracking
                    predictions = []
                    correct_count = 0
                    
                    with Progress() as progress:
                        task = progress.add_task(
                            f"Classifying with {model_name}", 
                            total=len(valid_examples)
                        )
                        
                        for example in valid_examples:
                            prediction = classifier.predict_multi(example.text)
                            predictions.append(prediction)
                            # A prediction is correct if all labels match exactly
                            if set(prediction.labels) == set(example.expected_labels):
                                correct_count += 1
                            progress.update(task, advance=1)
                    
                    # Calculate accuracy
                    accuracy = correct_count / len(valid_examples) if valid_examples else 0
                    
                    # Build prediction details
                    prediction_details = []
                    for example, pred in zip(valid_examples, predictions):
                        prediction_details.append({
                            "text": example.text,
                            "expected": example.expected_labels,
                            "predicted": pred.labels,
                        })
                    
                    # Build result
                    result = {
                        "model": model_name,
                        "eval_set_name": eval_set.name,
                        "total_examples": len(valid_examples),
                        "correct_predictions": correct_count,
                        "accuracy": accuracy,
                        "predictions": prediction_details,
                    }
                
                # Store results
                self.results_by_model[model_name][eval_set.name] = result
                
                # Track overall accuracy for cost-efficiency analysis
                if model_name not in self.accuracy_by_model:
                    self.accuracy_by_model[model_name] = {}
                self.accuracy_by_model[model_name][eval_set.name] = accuracy
                
                # Display accuracy for this evaluation
                self.console.print(f"    [green]Accuracy: {accuracy:.2%}[/green]")
        
        # Run all analysis steps
        self._run_analysis()
    
    def _run_analysis(self) -> None:
        """
        Run comprehensive analysis on the evaluation results.
        
        This includes:
        - Cost and latency analysis
        - Statistical bootstrapping for confidence intervals
        - Confusion matrix and error pattern analysis
        - Generating visualizations and reports
        """
        self.console.print("\n[bold cyan]Running Comprehensive Analysis...[/bold cyan]")
        
        # Run cost and latency analysis
        self.console.print("\n[bold green]Cost and Latency Analysis[/bold green]")
        costs = self.cost_analyzer.calculate_costs()
        latency_stats = self.cost_analyzer.calculate_latency_stats()
        
        # Display results
        self.cost_analyzer.display_cost_analysis(costs)
        self.cost_analyzer.display_latency_analysis(latency_stats)
        
        # Generate visualizations
        viz_dir = os.path.join(self.output_dir, "visualizations")
        
        # Latency distribution
        latency_viz = os.path.join(viz_dir, f"latency_distribution_{self.timestamp}.png")
        self.cost_analyzer.plot_latency_distribution(latency_viz)
        
        # Weighted average accuracy across all eval sets
        avg_accuracy = {}
        for model, eval_accuracies in self.accuracy_by_model.items():
            total_examples = 0
            weighted_sum = 0
            for eval_name, accuracy in eval_accuracies.items():
                result = self.results_by_model[model][eval_name]
                examples = result["total_examples"]
                total_examples += examples
                weighted_sum += accuracy * examples
            
            avg_accuracy[model] = weighted_sum / total_examples if total_examples > 0 else 0
        
        # Cost efficiency
        efficiency_viz = os.path.join(viz_dir, f"cost_efficiency_{self.timestamp}.png")
        self.cost_analyzer.plot_cost_efficiency(costs, avg_accuracy, efficiency_viz)
        
        # Save cost analysis results
        cost_output = os.path.join(self.output_dir, "metrics", f"cost_latency_{self.timestamp}.json")
        with open(cost_output, "w") as f:
            json.dump({
                "costs": costs,
                "latency": latency_stats,
                "avg_accuracy": avg_accuracy
            }, f, indent=2)
        
        # Run bootstrapped confidence interval analysis
        self.console.print("\n[bold green]Confidence Interval Analysis[/bold green]")
        bootstrap_results = {}
        
        for model_name, eval_results in self.results_by_model.items():
            bootstrap_results[model_name] = {}
            
            for eval_name, result in eval_results.items():
                # Run bootstrap analysis with our updated analyzer that handles dictionary results
                eval_bootstrap = self.bootstrap_analyzer.analyze(result)
                bootstrap_results[model_name][eval_name] = eval_bootstrap
                
                # Display results
                self.console.print(f"\n[bold]Model: {model_name}, Evaluation: {eval_name}[/bold]")
                self.bootstrap_analyzer.display_results(eval_bootstrap)
                
                # Generate visualizations
                bootstrap_viz = os.path.join(
                    viz_dir, 
                    f"bootstrap_{model_name}_{eval_name.replace(' ', '_')}_{self.timestamp}.png"
                )
                self.bootstrap_analyzer.plot_bootstrap_distributions(eval_bootstrap, bootstrap_viz)
        
        # Save bootstrap analysis results
        bootstrap_output = os.path.join(self.output_dir, "analysis", f"bootstrap_{self.timestamp}.json")
        with open(bootstrap_output, "w") as f:
            json.dump(bootstrap_results, f, indent=2)
        
        # Run confusion analysis
        self.console.print("\n[bold green]Confusion and Error Pattern Analysis[/bold green]")
        confusion_results = {}
        
        for model_name, eval_results in self.results_by_model.items():
            confusion_results[model_name] = {}
            
            for eval_name, result in eval_results.items():
                # Run confusion analysis with our updated analyzer that handles dictionary results
                confusion_analysis = self.confusion_analyzer.analyze(result)
                confusion_results[model_name][eval_name] = confusion_analysis
                
                if confusion_analysis:
                    # Display results
                    self.console.print(f"\n[bold]Model: {model_name}, Evaluation: {eval_name}[/bold]")
                    self.confusion_analyzer.display_results(confusion_analysis)
                    
                    # Generate visualizations
                    cm_viz = os.path.join(
                        viz_dir, 
                        f"confusion_matrix_{model_name}_{eval_name.replace(' ', '_')}_{self.timestamp}.png"
                    )
                    self.confusion_analyzer.plot_confusion_matrix(confusion_analysis, cm_viz)
                    
                    error_viz = os.path.join(
                        viz_dir, 
                        f"error_distribution_{model_name}_{eval_name.replace(' ', '_')}_{self.timestamp}.png"
                    )
                    self.confusion_analyzer.plot_error_distribution(confusion_analysis, error_viz)
        
        # Save confusion analysis results
        confusion_output = os.path.join(self.output_dir, "analysis", f"confusion_{self.timestamp}.json")
        with open(confusion_output, "w") as f:
            json.dump(confusion_results, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self) -> None:
        """Generate a comprehensive summary report of all evaluation results."""
        summary_table = Table(title="Model Performance Summary")
        summary_table.add_column("Evaluation Set", style="cyan")
        
        # Add columns for each model
        for model in self.models:
            summary_table.add_column(model, style="green")
        
        # Add a "best model" column
        summary_table.add_column("Best Model", style="yellow")
        
        # Add rows for each eval set
        for eval_set in self.evaluation_sets:
            row = [eval_set.name]
            
            # Get accuracy for each model
            accuracies = {}
            for model in self.models:
                if model in self.accuracy_by_model and eval_set.name in self.accuracy_by_model[model]:
                    accuracy = self.accuracy_by_model[model][eval_set.name]
                    accuracies[model] = accuracy
                    row.append(f"{accuracy:.2%}")
                else:
                    row.append("N/A")
            
            # Determine best model
            if accuracies:
                best_model = max(accuracies, key=accuracies.get)
                row.append(best_model)
            else:
                row.append("N/A")
            
            summary_table.add_row(*row)
        
        # Add a row for average accuracy
        avg_row = ["Weighted Average"]
        for model in self.models:
            total_examples = 0
            weighted_sum = 0
            for eval_name, accuracy in self.accuracy_by_model.get(model, {}).items():
                result = self.results_by_model[model][eval_name]
                examples = result["total_examples"]
                total_examples += examples
                weighted_sum += accuracy * examples
            
            avg = weighted_sum / total_examples if total_examples > 0 else 0
            avg_row.append(f"{avg:.2%}")
        
        # Add best model for average
        avg_accuracies = {}
        for model in self.models:
            total_examples = 0
            weighted_sum = 0
            for eval_name, accuracy in self.accuracy_by_model.get(model, {}).items():
                result = self.results_by_model[model][eval_name]
                examples = result["total_examples"]
                total_examples += examples
                weighted_sum += accuracy * examples
            
            avg = weighted_sum / total_examples if total_examples > 0 else 0
            avg_accuracies[model] = avg
        
        if avg_accuracies:
            best_model = max(avg_accuracies, key=avg_accuracies.get)
            avg_row.append(best_model)
        else:
            avg_row.append("N/A")
        
        summary_table.add_row(*avg_row)
        
        # Display summary table
        self.console.print("\n[bold green]Evaluation Summary[/bold green]")
        self.console.print(summary_table)
        
        # Calculate cost efficiency
        costs = self.cost_analyzer.calculate_costs()
        efficiency_table = Table(title="Cost Efficiency Analysis")
        efficiency_table.add_column("Model", style="cyan")
        efficiency_table.add_column("Avg. Accuracy", style="green")
        efficiency_table.add_column("Total Cost", style="yellow")
        efficiency_table.add_column("Accuracy/$ Ratio", style="red")
        
        for model in self.models:
            avg_accuracy = avg_accuracies.get(model, 0)
            total_cost = costs.get(model, {}).get("total_cost", None)
            
            if total_cost:
                efficiency = avg_accuracy / total_cost
                efficiency_table.add_row(
                    model,
                    f"{avg_accuracy:.2%}",
                    f"${total_cost:.4f}",
                    f"{efficiency:.2%}/$"
                )
            else:
                efficiency_table.add_row(
                    model,
                    f"{avg_accuracy:.2%}",
                    "N/A",
                    "N/A"
                )
        
        self.console.print("\n")
        self.console.print(efficiency_table)
        
        # Save summary report
        summary_output = os.path.join(self.output_dir, f"summary_report_{self.timestamp}.txt")
        with open(summary_output, "w") as f:
            f.write("Classification Evaluation Summary\n")
            f.write("===============================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models: {', '.join(self.models)}\n")
            f.write(f"Evaluation Sets: {', '.join(es.name for es in self.evaluation_sets)}\n\n")
            
            f.write("Performance Summary:\n")
            for model in self.models:
                f.write(f"\n{model}:\n")
                for eval_name, accuracy in self.accuracy_by_model.get(model, {}).items():
                    f.write(f"  {eval_name}: {accuracy:.2%}\n")
                
                f.write(f"  Average: {avg_accuracies.get(model, 0):.2%}\n")
            
            f.write("\nCost Analysis:\n")
            for model in self.models:
                if model in costs:
                    cost = costs[model]
                    f.write(f"\n{model}:\n")
                    f.write(f"  Input Tokens: {cost['input_tokens']:,}\n")
                    f.write(f"  Output Tokens: {cost['output_tokens']:,}\n")
                    f.write(f"  Total Cost: ${cost['total_cost']:.4f}\n")
                    
                    avg_accuracy = avg_accuracies.get(model, 0)
                    efficiency = avg_accuracy / cost['total_cost'] if cost['total_cost'] > 0 else 0
                    f.write(f"  Efficiency: {efficiency:.2%}/$ (higher is better)\n")
            
            f.write("\nRecommendation:\n")
            best_accuracy_model = max(avg_accuracies, key=avg_accuracies.get) if avg_accuracies else None
            
            if best_accuracy_model:
                f.write(f"- Best accuracy: {best_accuracy_model} ({avg_accuracies[best_accuracy_model]:.2%})\n")
            
            # Find best efficiency
            best_efficiency = None
            best_efficiency_model = None
            
            for model in self.models:
                if model in costs and costs[model]['total_cost'] > 0:
                    efficiency = avg_accuracies.get(model, 0) / costs[model]['total_cost']
                    if best_efficiency is None or efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_efficiency_model = model
            
            if best_efficiency_model:
                f.write(f"- Best efficiency: {best_efficiency_model} ({best_efficiency:.2%}/$ ratio)\n")
        
        self.console.print(f"\n[green]Summary report saved to: {summary_output}[/green]")
        
        # Save all detailed results
        for model_name, eval_results in self.results_by_model.items():
            for eval_name, result in eval_results.items():
                result_output = os.path.join(
                    self.output_dir, 
                    "metrics", 
                    f"{model_name}_{eval_name.replace(' ', '_')}_{self.timestamp}.json"
                )
                with open(result_output, "w") as f:
                    json.dump(result, f, indent=2)


def main():
    """Main entry point for the unified evaluation framework."""
    parser = argparse.ArgumentParser(description="Unified classification evaluation framework")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the evaluation configuration YAML file"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize and run evaluator
        evaluator = UnifiedEvaluator(args.config)
        evaluator.prepare()
        evaluator.run()
        
        # Success message
        Console().print("\n[bold green]Evaluation completed successfully![/bold green]")
        
    except KeyboardInterrupt:
        Console().print("\n[bold yellow]Evaluation cancelled by user.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        Console().print(f"\n[bold red]Error during evaluation:[/bold red] {str(e)}")
        import traceback
        Console().print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()