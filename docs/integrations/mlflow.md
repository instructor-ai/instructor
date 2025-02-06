# MLflow Instructor Tracing and Model Logging

[MLflow](https://mlflow.org/docs/latest/introduction/index.html) is an open-source platform for managing the machine learning lifecycle, with comprehensive support for GenAI applications including model deployment, prompt engineering, and native integrations with popular LLM providers.

## MLflow Tracing with Instructor

MLflow can provide detailed [tracing](https://mlflow.org/docs/latest/llms/tracing/index.html) of LLM interactions made with Instructor. MLflow tracing integrates with Instructor via the underlying LLM client libraries, enabling automatic logging of all LLM interactions and structured outputs with a simple call to `mlflow.<provider>.autolog()`.

### OpenAI Tracing Example

Here's a simple example showing how to enable tracing with OpenAI. To get started, first install the Instructor, MLflow, and OpenAI Python packages:

```bash
pip install instructor mlflow openai
```

Then, enable tracing with `mlflow.openai.autolog()`. Afterward, all calls to OpenAI models, including those made with Instructor, will be captured as MLflow traces. Read [this guide](https://mlflow.org/docs/latest/tracking/autolog.html) for more on getting started with MLflow autologging.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator
import mlflow

# Set up MLflow tracking
mlflow.set_experiment("mlflow-instructor")
mlflow.openai.autolog()

# Create Instructor client
client = instructor.from_openai(OpenAI())

class EmailSubject(BaseModel):
    recipient: str
    subject: Annotated[
        str,
        BeforeValidator(
            llm_validator(
                "ensure subject is concise and engaging",
                client=client,
                model="gpt-4o"
            )
        )
    ]

response = client.chat.completions.create(
    model="gpt-4o",
    response_model=EmailSubject,
    messages=[{
        "role": "user",
        "content": "Come up with a subject for an email inviting users to follow the MLflow GitHub repo."
    }]
)
```

Now, start the MLflow UI with `mlflow ui`. You'll be able to view the traces in the UI by clicking on the "mlflow-instructor" experiment and then clicking on the "Traces" tab.

![MLflow Tracing UI](/img/mlflow_tracing_ui.png)

### Anthropic Tracing Example

As noted above, MLflow autologging works with Instructor by tracing the client library calls. Here we'll adapt the above example to use an Anthropic model.

```python
import anthropic
import instructor
import mlflow

# Enable tracing for Anthropic
mlflow.anthropic.autolog()

# Create Instructor client for Anthropic
client = instructor.from_anthropic(anthropic.Anthropic())

# Use the same EmailSubject model
response = client.chat.completions.create(
    model="claude-3-sonnet-20240229",
    response_model=EmailSubject,
    messages=[{
        "role": "user",
        "content": "Come up with a subject for an email inviting users to follow the MLflow GitHub repo."
    }],
    max_tokens=1000
)
```

### What Gets Traced?

When using MLflow's autologging with Instructor, you get visibility into:

- All LLM API calls and their responses
- Model parameters and configurations
- Validation steps and retries
- Timing information
- Any other outputs returned by the client library

### Viewing Traces

Traces can be viewed in the MLflow UI, which provides a rich interface for:

- Exploring the full message history
- Analyzing model performance
- Debugging validation issues
- Monitoring costs and usage

## Related Resources

- [MLflow Tracing Documentation](https://mlflow.org/docs/latest/llms/tracing/index.html)

## Updates and Compatibility

MLflow tracing support is available in MLflow 2.14.0 and later. Always check the MLflow documentation for the latest compatibility information and features.
