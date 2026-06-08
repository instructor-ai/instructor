"""v2 Vertex AI client factory."""

from __future__ import annotations

import os
import warnings
from typing import Any, Literal, TYPE_CHECKING, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.client_factory import create_instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider

if TYPE_CHECKING:
    import vertexai.generative_models as gm
else:
    try:
        import vertexai.generative_models as gm
    except ImportError:
        gm = None


@overload
def from_vertexai(
    client: gm.GenerativeModel,
    mode: Mode = Mode.TOOLS,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_vertexai(
    client: gm.GenerativeModel,
    mode: Mode = Mode.TOOLS,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_vertexai(
    client: gm.GenerativeModel,
    mode: Mode = Mode.TOOLS,
    use_async: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    generative_model_type = getattr(gm, "GenerativeModel", None)
    sync_types = (
        (generative_model_type,) if isinstance(generative_model_type, type) else None
    )

    return create_instructor(
        client,
        provider=Provider.VERTEXAI,
        mode=mode,
        use_async=use_async,
        sync_types=sync_types,
        **kwargs,
    )


def build_from_model(
    *,
    provider: Provider,  # noqa: ARG001
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,  # noqa: ARG001
    kwargs: dict[str, Any],
) -> Instructor | AsyncInstructor:
    from instructor.v2.core.errors import ConfigurationError

    warnings.warn(
        "The 'vertexai' provider is deprecated. Use 'google' provider with "
        "vertexai=True instead. Example: "
        "instructor.from_provider('google/gemini-pro', vertexai=True)",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        import vertexai
    except ImportError:
        raise ConfigurationError(
            "The vertexai package is required to use the VertexAI provider. "
            "Install it with `pip install google-cloud-aiplatform`."
        ) from None
    if gm is None:
        raise ConfigurationError(
            "The vertexai package is required to use the VertexAI provider. "
            "Install it with `pip install google-cloud-aiplatform`."
        )
    project = kwargs.pop("project", os.environ.get("GOOGLE_CLOUD_PROJECT"))
    if not project:
        raise ValueError(
            "Project ID is required for Vertex AI. Set it with "
            "`export GOOGLE_CLOUD_PROJECT=<your-project-id>` or pass it as "
            "kwarg project=<your-project-id>"
        )
    vertexai.init(
        project=project,
        location=kwargs.pop(
            "location", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        ),
        credentials=kwargs.pop("credentials", None),
    )
    return from_vertexai(
        gm.GenerativeModel(model_name),
        use_async=async_client,
        mode=mode or Mode.TOOLS,
        **kwargs,
    )


__all__ = ["build_from_model", "from_vertexai"]
