# Unified Classification Evaluation Harness

A comprehensive framework for evaluating text classification models built with Instructor and OpenAI. This toolkit provides detailed metrics on accuracy, cost-efficiency, and error patterns to help you select the optimal model for your classification tasks.

## Features

- **Multi-model Evaluation**: Compare performance across different LLM models
- **Comprehensive Metrics**: Get detailed accuracy, precision, recall, and F1 scores
- **Cost & Latency Analysis**: Track token usage, costs, and response times
- **Statistical Reliability**: Bootstrapped confidence intervals for accuracy metrics
- **Error Analysis**: Detailed confusion matrices and error pattern detection
- **Rich Visualizations**: Charts and tables for all key metrics
- **Unified Workflow**: Streamlined process from evaluation to analysis

## Directory Structure

```
eval_harness/
├── configs/                # Configuration files
│   ├── unified_config.yaml # Main unified configuration
│   ├── test_config.yaml    # Quick test configuration
│   └── unified_results/    # Results output directory
│       ├── analysis/       # Detailed analysis outputs
│       ├── metrics/        # Performance metrics
│       └── visualizations/ # Charts and graphs
├── datasets/               # Evaluation datasets
│   ├── custom_evalset.yaml # Single-label evaluation examples
│   ├── complex_evalset.yaml # Challenging evaluation examples
│   ├── multi_label_evalset.yaml # Multi-label examples
│   └── test_evalset.yaml   # Minimal test dataset
├── utils/                  # Utility modules
│   └── analysis.py         # Analysis toolkit with metrics, visualizations
└── unified_eval.py         # Main unified evaluation script
```

## Getting Started

### Prerequisites

1. Install the required dependencies:

```bash
pip install instructor openai pydantic rich pyyaml pandas matplotlib tabulate scikit-learn seaborn
```

2. Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. Ensure you have a classification definition file (see format below)

### Running Unified Evaluation

The simplest way to run a complete evaluation:

```bash
python unified_eval.py --config configs/unified_config.yaml
```

For quick testing and development with a minimal dataset:

```bash
python unified_eval.py --config configs/test_config.yaml
```

The evaluation process:
1. Loads models and datasets specified in the config
2. Runs evaluations on all combinations
3. Tracks accuracy, cost, and latency metrics
4. Generates statistical analysis with confidence intervals
5. Analyzes error patterns and confusion matrices
6. Creates visualizations for all metrics
7. Produces a comprehensive summary report

## Configuration

The unified configuration file (`unified_config.yaml`) controls all aspects of the evaluation:

```yaml
# Models to evaluate
models:
  - "gpt-3.5-turbo"  # Fast, low-cost model
  - "gpt-4o-mini"    # Mid-tier model with good performance

# Classification definition file (details below)
definition_path: "../../tests/intent_classification.yaml"

# Evaluation sets to run (full paths or relative to the harness)
eval_sets:
  - "../datasets/custom_evalset.yaml"
  - "../datasets/complex_evalset.yaml"

# Analysis parameters
bootstrap_samples: 1000     # Number of bootstrap resamples for confidence intervals
confidence_level: 0.95      # Confidence level (e.g., 95%)

# Parallel processing 
n_jobs: 4                   # Number of parallel jobs for evaluation

# Output directory for results
output_dir: "unified_results"  # Directory to store results
```

## Classification Definition

Classification definitions are stored in YAML files with this format:

```yaml
name: "Intent Classification"
description: "Classify user requests into intent categories"

# For single-label classification
output_class:
  name: "IntentClassification"
  description: "Classifies user intent into categories"
  definitions:
    label:
      type: "string"
      description: "The intent category"
      enum:
        - "question"    # Questions seeking information
        - "scheduling"  # Requests to schedule or plan events
        - "coding"      # Questions about coding or programming
```

For multi-label classification:

```yaml
# For multi-label classification
output_class:
  name: "MultiLabelIntentClassification"
  description: "Classifies user intent into multiple categories"
  definitions:
    labels:
      type: "array"
      description: "The intent categories that apply"
      items:
        type: "string"
        enum:
          - "question"
          - "scheduling"
          - "coding"
```

## Evaluation Sets

Evaluation sets are defined in YAML files following this format:

For single-label classification:

```yaml
name: "Custom Classification Evaluation Set"
description: "Custom evaluation set for the intent classification"
classification_type: "single"
examples:
  - text: "What's the difference between Python 2 and Python 3?"
    expected_label: "question"
  
  - text: "Book a team lunch for next Tuesday at noon."
    expected_label: "scheduling"
  
  - text: "Help me optimize this database query for better performance."
    expected_label: "coding"
```

For multi-label classification:

```yaml
name: "Multi-label Evaluation Set"
description: "Test set for multi-label classification"
classification_type: "multi"
examples:
  - text: "Help me schedule a coding session."
    expected_labels: ["scheduling", "coding"]
    
  - text: "How do I code an efficient scheduling algorithm?"
    expected_labels: ["question", "coding"]
```

## Output Analysis

### Performance Metrics

The framework generates detailed metrics for each model and dataset:

- **Accuracy**: Percentage of correctly classified examples
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Bootstrapped Confidence Intervals**: Statistical bounds on all metrics

### Cost Analysis

Detailed cost tracking for each model:

- **Token Usage**: Input and output token counts
- **API Costs**: Calculated expenses based on current pricing
- **Cost Efficiency**: Accuracy-to-cost ratios for value comparison
- **Latency Metrics**: Response time statistics (min, mean, median, P90, P95)

### Error Analysis

Analysis of classification mistakes:

- **Confusion Matrix**: Detailed breakdown of prediction patterns
- **Error Distribution**: Where models struggle most frequently
- **Error Patterns**: Common features in misclassified examples
- **Error Examples**: Specific examples that were misclassified

## Example Output

The framework outputs a comprehensive summary report like this:

```
Classification Evaluation Summary
===============================

Date: 2025-04-18 17:59:19
Models: gpt-3.5-turbo, gpt-4o-mini
Evaluation Sets: Custom Classification Evaluation Set, Complex Classification Evaluation Set

Performance Summary:

gpt-3.5-turbo:
  Custom Classification Evaluation Set: 90.91%
  Complex Classification Evaluation Set: 100.00%
  Average: 96.15%

gpt-4o-mini:
  Custom Classification Evaluation Set: 100.00%
  Complex Classification Evaluation Set: 100.00%
  Average: 100.00%

Cost Analysis:

gpt-3.5-turbo:
  Input Tokens: 24,659
  Output Tokens: 164
  Total Cost: $0.0126
  Efficiency: 7646.13%/$ (higher is better)

gpt-4o-mini:
  Input Tokens: 24,415
  Output Tokens: 164
  Total Cost: $0.0376
  Efficiency: 2659.11%/$ (higher is better)

Recommendation:
- Best accuracy: gpt-4o-mini (100.00%)
- Best efficiency: gpt-3.5-turbo (7646.13%/$ ratio)
```

## Advanced Usage

### Creating Custom Evaluation Sets

To create a custom evaluation set:

1. Create a new YAML file in the `datasets/` directory
2. Follow the format in the examples above
3. Add the path to your config file
4. Run the evaluation with your config

### Extending Analysis

To add new analysis techniques:

1. Extend the relevant classes in `utils/analysis.py`
2. Update the unified evaluation script to use your new analysis
3. Ensure any new visualizations are properly saved

## Best Practices

- **Balanced Datasets**: Create evaluation sets with balanced class distributions
- **Real-world Examples**: Include challenging, ambiguous examples that represent real user inputs
- **Multiple Models**: Compare at least 2-3 different models for reliable comparisons
- **Cost-conscious Testing**: Start with the test configuration for quick iterations

---

This evaluation harness empowers data scientists and ML engineers to make informed, data-driven decisions about which classification models will best serve their applications, balancing accuracy, cost, and performance.