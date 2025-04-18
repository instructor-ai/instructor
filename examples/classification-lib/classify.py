from __future__ import annotations

from typing import Any
from pydantic import BaseModel
import instructor
from openai.types.chat import ChatCompletionMessageParam
import asyncio
import multiprocessing

# ------------------------------------------------------------------
# Optional progress bar support (requires ``tqdm``)
# ------------------------------------------------------------------
# ``tqdm`` is a lightweight dependency widely used for progress bars. If it is
# not available in the runtime environment, we gracefully fall back to no‑op
# shims so that the rest of the code keeps working.

try:
    from tqdm import tqdm  # type: ignore
    from tqdm.asyncio import tqdm as tqdm_async  # type: ignore

except ModuleNotFoundError:  # pragma: no cover – allow running without tqdm
    # Fallbacks that mimic the minimal tqdm API we rely on (iteration + gather)
    def _identity(iterable=None, *_, **__):  # noqa: D401 – simple helper
        """Return *iterable* unchanged – replacement for :pyclass:`tqdm.tqdm`."""

        if iterable is None:
            iterable = []
        return iterable

    class _AsyncTqdmShim:  # pylint: disable=too-few-public-methods
        """Subset of ``tqdm.asyncio.tqdm`` we need (only ``gather``)."""

        @classmethod
        async def gather(cls, *coros, **_):  # type: ignore[no-self-use]
            import asyncio as _asyncio  # local import to avoid top‑level dep

            return await _asyncio.gather(*coros)

    tqdm = _identity  # type: ignore  # pylint: disable=invalid-name
    tqdm_async = _AsyncTqdmShim  # type: ignore  # pylint: disable=invalid-name


from .schema import ClassificationDefinition


class Classifier:
    """
    A fluent API for classification using OpenAI and Instructor.

    """

    def __init__(self, classification_definition: ClassificationDefinition):
        """Initialize a classifier with a classification definition."""
        self.definition = classification_definition
        self.client: instructor.Instructor | instructor.AsyncInstructor | None = None
        self.model_name: str | None = None

        # Dynamically generated pydantic models for single‑ and multi‑label predictions
        self._classification_model: type[BaseModel] = (
            self.definition.get_classification_model()
        )
        self._multi_classification_model: type[BaseModel] = (
            self.definition.get_multiclassification_model()
        )

    def with_client(self, client: instructor.Instructor):
        """Attach an OpenAI client (wrapped by Instructor) and return self."""
        self.client = client
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

    def _validate_client(self):
        """Validate that client is set and appropriate for the method type."""
        if not self.client:
            raise ValueError("Client not set. Use `.with_client()` first.")
        if isinstance(self.client, instructor.AsyncInstructor):
            raise ValueError(
                "AsyncInstructor cannot be used with synchronous methods. Use AsyncClassifier instead."
            )

    def predict(self, text: str) -> BaseModel:
        """Single‑label prediction – returns an instance of the generated model ``T``."""
        self._validate_client()

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
        self._validate_client()

        messages, context = self._build_messages(text)
        result = self.client.chat.completions.create(
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )
        return result

    # ------------------------------------------------------------------
    # Synchronous batch prediction
    # ------------------------------------------------------------------

    def batch_predict(
        self, texts: list[str], n_jobs: int | None = None
    ) -> list[BaseModel]:
        """Run :py:meth:`predict` over multiple texts in parallel."""
        self._validate_client()

        if not texts:
            return []

        # Determine desired parallelism level (defaults to ``cpu_count``)
        if n_jobs is None:
            cnt = multiprocessing.cpu_count() or 1
            n_jobs = min(len(texts), cnt)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            it = executor.map(self.predict, texts)
            results = list(tqdm(it, total=len(texts), desc="classify", leave=False))
        return results

    def batch_predict_multi(
        self, texts: list[str], n_jobs: int | None = None
    ) -> list[BaseModel]:
        """Run :py:meth:`predict_multi` over multiple texts in parallel.

        The same strategy selection as :py:meth:`batch_predict` applies –
        threads are used when a custom client is attached, otherwise the
        original *multiprocessing* implementation is retained.
        """
        self._validate_client()

        if not texts:
            return []

        if n_jobs is None:
            cnt = multiprocessing.cpu_count() or 1
            n_jobs = min(len(texts), cnt)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            it = executor.map(self.predict_multi, texts)
            results = list(
                tqdm(it, total=len(texts), desc="classify‑multi", leave=False)
            )
        return results

    def predict_with_completion(self, text: str) -> tuple[BaseModel, Any]:
        """Return both the parsed model and the underlying LLM completion."""
        self._validate_client()

        messages, context = self._build_messages(text)
        return self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )

    def predict_multi_with_completion(self, text: str) -> tuple[BaseModel, Any]:
        """Multi‑label variant with raw completion."""
        self._validate_client()

        messages, context = self._build_messages(text)
        return self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )

    def batch_predict_with_completion(
        self, texts: list[str], n_jobs: int | None = None
    ) -> list[tuple[BaseModel, Any]]:
        raise NotImplementedError(
            "Batch prediction with completion is not implemented."
        )

    def batch_predict_multi_with_completion(
        self, texts: list[str], n_jobs: int | None = None
    ) -> list[tuple[BaseModel, Any]]:
        raise NotImplementedError(
            "Batch prediction with completion is not implemented."
        )


class AsyncClassifier(Classifier):
    """Asynchronous variant of :class:`Classifier`. All prediction methods are
    defined as *coroutines* and must be awaited.

    The constructor and fluent helpers are inherited from :class:`Classifier` –
    you can therefore use the same API:

    >>> classifier = AsyncClassifier(definition).with_client(AsyncOpenAI())
    >>> result = await classifier.predict("example text")
    """

    def with_client(self, client: instructor.AsyncInstructor):
        """Attach an OpenAI client (wrapped by Instructor) and return self."""
        self.client = client
        return self

    def _validate_async_client(self):
        """Validate that async client is set and ready for use."""
        if not self.client or not self.model_name:
            raise ValueError(
                "Client and model name must be set. Use `.with_client()` and `.with_model()` first."
            )

    async def predict(self, text: str) -> BaseModel:  # type: ignore[override]
        """Asynchronously predict a single label for *text*."""
        self._validate_async_client()

        messages, context = self._build_messages(text)
        return await self.client.chat.completions.create(  # type: ignore[return-value]
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )

    async def predict_multi(self, text: str) -> BaseModel:  # type: ignore[override]
        """Asynchronously predict multiple labels for *text*."""
        self._validate_async_client()

        messages, context = self._build_messages(text)
        return await self.client.chat.completions.create(  # type: ignore[return-value]
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )

    # ------------------------------------------------------------------
    # Async batch prediction with concurrency control
    # ------------------------------------------------------------------

    async def batch_predict(
        self, texts: list[str], *, n_jobs: int = 5
    ) -> list[BaseModel]:
        """Run :py:meth:`predict` concurrently over *texts* using a semaphore.

        Parameters
        ----------
        texts:
            The list of input strings to classify.
        n_jobs:
            Maximum number of concurrent classification jobs.
        """
        self._validate_async_client()
        sem = asyncio.Semaphore(n_jobs)

        async def _worker(t: str):
            async with sem:
                return await self.predict(t)

        tasks = [_worker(t) for t in texts]

        # If tqdm is available, leverage its asyncio integration for a neat
        # progress bar. Otherwise, fall back to plain ``asyncio.gather``.

        try:
            return await tqdm_async.gather(*tasks, total=len(tasks), desc="classify")
        except Exception:  # pragma: no cover – safety net if tqdm absent
            return await asyncio.gather(*tasks)

    async def batch_predict_multi(
        self, texts: list[str], *, n_jobs: int = 5
    ) -> list[BaseModel]:
        """Run :py:meth:`predict_multi` concurrently over *texts* using a semaphore."""
        self._validate_async_client()
        sem = asyncio.Semaphore(n_jobs)

        async def _worker(t: str):
            async with sem:
                return await self.predict_multi(t)

        tasks = [_worker(t) for t in texts]

        try:
            return await tqdm_async.gather(
                *tasks, total=len(tasks), desc="classify‑multi"
            )
        except Exception:  # pragma: no cover
            return await asyncio.gather(*tasks)

    async def predict_with_completion(self, text: str) -> tuple[BaseModel, Any]:
        self._validate_async_client()

        messages, context = self._build_messages(text)
        model, completion = await self.client.chat.completions.create_with_completion(  # type: ignore[call-arg]
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )
        return model, completion

    async def predict_multi_with_completion(self, text: str) -> tuple[BaseModel, Any]:
        self._validate_async_client()

        messages, context = self._build_messages(text)
        model, completion = await self.client.chat.completions.create_with_completion(  # type: ignore[call-arg]
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )
        return model, completion

    async def batch_predict_with_completion(
        self, texts: list[str], *, n_jobs: int = 5
    ) -> list[tuple[BaseModel, Any]]:
        sem = asyncio.Semaphore(n_jobs)

        async def _worker(t: str):
            async with sem:
                return await self.predict_with_completion(t)

        tasks = [_worker(t) for t in texts]

        try:
            return await tqdm_async.gather(*tasks, total=len(tasks), desc="classify+")
        except Exception:  # pragma: no cover
            return await asyncio.gather(*tasks)

    async def batch_predict_multi_with_completion(
        self, texts: list[str], *, n_jobs: int = 5
    ) -> list[tuple[BaseModel, Any]]:
        sem = asyncio.Semaphore(n_jobs)

        async def _worker(t: str):
            async with sem:
                return await self.predict_multi_with_completion(t)

        tasks = [_worker(t) for t in texts]

        try:
            return await tqdm_async.gather(
                *tasks, total=len(tasks), desc="classify‑multi+"
            )
        except Exception:  # pragma: no cover
            return await asyncio.gather(*tasks)
