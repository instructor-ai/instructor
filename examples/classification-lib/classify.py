from typing import Any
from pydantic import BaseModel
import instructor
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# Import from local module with relative import
from schema import ClassificationDefinition


class Classifier:
    """
    A fluent API for classification using OpenAI and Instructor.

    """

    def __init__(self, classification_definition: ClassificationDefinition):
        """Initialize a classifier with a classification definition."""
        self.definition = classification_definition
        self.client: Any = None
        self.model_name: str = "gpt-4o-mini"

        # Dynamically generated pydantic models for single‑ and multi‑label predictions
        self._classification_model: type[BaseModel] = (
            self.definition.get_classification_model()
        )
        self._multi_classification_model: type[BaseModel] = (
            self.definition.get_multiclassification_model()
        )

    def with_client(self, client: OpenAI):
        """Attach an OpenAI client (wrapped by Instructor) and return self."""
        self.client = instructor.from_openai(client)
        return self

    def with_model(self, model_name: str):
        """Specify which model to use (e.g., ``gpt-4o``) and return self."""
        self.model_name = model_name
        return self

    def _build_messages(
        self, text: str, include_examples: bool = True
    ) -> tuple[list[ChatCompletionMessageParam], dict[str, Any]]:
        """Construct chat messages using Jinja templating.

        This leverages Instructor's built‑in Jinja support (see docs/concepts/templating.md).
        The returned tuple contains the messages *and* the rendering context which
        should be forwarded to ``client.chat.completions.create``.

        Parameters
        ----------
        text:
            The text to classify.
        include_examples:
            Toggle whether few‑shot examples should be embedded in the prompt. This
            demonstrates the use of Jinja conditionals.
        """

        messages: list[ChatCompletionMessageParam] = []

        # ------------------------------------------------------------------
        # Optional system message
        # ------------------------------------------------------------------
        if self.definition.system_message:
            messages.append(
                {
                    "role": "system",
                    "content": self.definition.system_message,
                }
            )

        # ------------------------------------------------------------------
        # User message template – relies on Jinja to render examples and text
        # ------------------------------------------------------------------
        user_template = """
{% if include_examples %}
<examples>
{% for ld in label_definitions %}
  <label name="{{ ld.label }}">
    <description>{{ ld.description }}</description>
    {% if ld.examples and ld.examples.examples_positive %}
      {% for ex in ld.examples.examples_positive %}
      <example type="positive" label="{{ ld.label }}">{{ ex }}</example>
      {% endfor %}
    {% endif %}
    {% if ld.examples and ld.examples.examples_negative %}
      {% for ex in ld.examples.examples_negative %}
      <example type="negative" label="{{ ld.label }}">{{ ex }}</example>
      {% endfor %}
    {% endif %}
  </label>
{% endfor %}
</examples>
{% endif %}

<classify>
  Classify the following text into one of the following labels:
  <labels>
  {% for ld in label_definitions %}
    <label name="{{ ld.label }}">{{ ld.label }}</label>
  {% endfor %}
  </labels>
  <text>{{ input_text }}</text>
</classify>
            """

        messages.append({"role": "user", "content": user_template})

        # ------------------------------------------------------------------
        # Build the Jinja rendering context
        # ------------------------------------------------------------------
        context = {
            "label_definitions": self.definition.label_definitions,  # keep original objects accessible
            "input_text": text,
            "include_examples": include_examples,
        }

        return messages, context

    def predict(self, text: str) -> BaseModel:
        """Single‑label prediction – returns an instance of the generated model ``T``."""
        if not self.client:
            raise ValueError("Client not set. Use `.with_client()` first.")

        messages, context = self._build_messages(text)
        result = self.client.chat.completions.create(
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )
        return result

    def predict_multi(self, text: str) -> BaseModel:
        """Multi‑label prediction – returns an instance of the generated model ``M``."""
        if not self.client:
            raise ValueError("Client not set. Use `.with_client()` first.")

        messages, context = self._build_messages(text)
        result = self.client.chat.completions.create(
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )
        return result
