import os
import sys
from jinja2 import Environment

# Ensure we can import modules from the parent directory (../)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from classify import Classifier  # type: ignore
from schema import ClassificationDefinition  # type: ignore


def _render_template(content: str, context: dict) -> str:
    """Utility to render a Jinja template string using a basic environment."""
    env = Environment()
    template = env.from_string(content)
    return template.render(**context)


def test_build_messages_conditionals():
    """Verify that the `_build_messages` method respects the `include_examples` flag."""

    # ------------------------------------------------------------------
    # Setup – load classification definition and instantiate the classifier
    # ------------------------------------------------------------------
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)
    classifier = Classifier(definition)

    # Text we want to classify
    sample_text = "Hello, how are you?"

    # ------------------------------------------------------------------
    # Case 1: include_examples=True (default)
    # ------------------------------------------------------------------
    msgs_with_ex, ctx_with_ex = classifier._build_messages(
        sample_text, include_examples=True
    )

    # The user message should render with examples present
    user_msg_with_ex = next(m for m in msgs_with_ex if m["role"] == "user")
    rendered_with_ex = _render_template(user_msg_with_ex["content"], ctx_with_ex)

    assert "<examples>" in rendered_with_ex, (
        "Expected <examples> block when include_examples is True"
    )
    assert sample_text in rendered_with_ex, (
        "Input text should be present in rendered prompt"
    )

    # ------------------------------------------------------------------
    # Case 2: include_examples=False
    # ------------------------------------------------------------------
    msgs_no_ex, ctx_no_ex = classifier._build_messages(
        sample_text, include_examples=False
    )

    user_msg_no_ex = next(m for m in msgs_no_ex if m["role"] == "user")
    rendered_no_ex = _render_template(user_msg_no_ex["content"], ctx_no_ex)

    assert "<examples>" not in rendered_no_ex, (
        "<examples> block should be omitted when flag is False"
    )
    assert sample_text in rendered_no_ex, (
        "Input text should still be present when examples omitted"
    )
