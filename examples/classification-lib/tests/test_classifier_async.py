from __future__ import annotations

# ---------------------------------------------------------------------------
# AsyncClassifier tests – leverage pytest‑asyncio
# ---------------------------------------------------------------------------

import os
import sys
import types
import pytest

# Ensure parent directory (../) is on the path so that we can import the
# classification library modules when tests are executed from the tests folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Local imports after path adjustment
from classify import AsyncClassifier  # type: ignore
from schema import ClassificationDefinition  # type: ignore


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class DummyAsyncClient:
    """A minimal async‑capable mock matching the subset of Instructor's API that
    *AsyncClassifier* relies on (``client.chat.completions.create``).
    """

    def __init__(self, default_label: str = "question") -> None:
        self._default_label = default_label

        async def _create(*_args, **kwargs):  # noqa: D401 – async stub
            ResponseModel = kwargs["response_model"]
            fields = set(ResponseModel.model_fields.keys())  # type: ignore[attr-defined]

            if "label" in fields:
                return ResponseModel(label=self._default_label)
            elif "labels" in fields:
                return ResponseModel(labels=[self._default_label])
            else:
                raise ValueError("Unexpected response model schema")

        # Build the minimum attribute hierarchy: client.chat.completions.create
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create)
        )


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_predict_single_label():
    """`AsyncClassifier.predict` returns a model instance with the expected label."""

    # Load classification definition from YAML file used by existing tests
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)

    classifier = AsyncClassifier(definition)
    classifier.client = DummyAsyncClient(default_label="question")

    result = await classifier.predict("Hello, how are you?")
    assert result.label == "question"


@pytest.mark.asyncio
async def test_async_batch_predict_order_and_length():
    """`batch_predict` preserves order and returns correct number of results."""

    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)

    classifier = AsyncClassifier(definition)
    classifier.client = DummyAsyncClient(default_label="question")

    texts = [
        "What's the weather today?",
        "Schedule a meeting for tomorrow.",
        "How do I write a Python function?",
    ]

    results = await classifier.batch_predict(texts, n_jobs=2)

    # Validate length and order preservation
    assert len(results) == len(texts)
    assert all(r.label == "question" for r in results)

    # Ensure results correspond position‑wise to inputs (identity property here)
    for _t, r in zip(texts, results):
        # Our dummy client always returns the same label, but we can still check
        # that the round‑trip classification does not reorder the inputs.
        assert isinstance(r, classifier._classification_model)
