"""v2 Bedrock client factory."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.client_factory import create_instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.bedrock import handlers  # noqa: F401

if TYPE_CHECKING:
    from botocore.client import BaseClient
else:
    try:
        from botocore.client import BaseClient
    except ImportError:
        BaseClient = None


@overload
def from_bedrock(
    client: BaseClient,
    mode: Mode = Mode.TOOLS,
    async_client: Literal[False] = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_bedrock(
    client: BaseClient,
    mode: Mode = Mode.TOOLS,
    async_client: Literal[True] = True,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_bedrock(
    client: BaseClient,
    mode: Mode = Mode.TOOLS,
    async_client: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Bedrock client using v2 registry.

    Bedrock uses the Converse API through a boto3 BaseClient. This factory supports
    TOOLS and MD_JSON modes, and can wrap calls in an async interface if needed.

    Args:
        client: boto3 Bedrock Runtime client
        mode: The mode to use (defaults to Mode.TOOLS)
        async_client: Whether to return an async Instructor wrapper
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on async_client)

    Raises:
        ModeError: If mode is not registered for Bedrock
        ClientError: If client is not a valid BaseClient or botocore not installed
    """
    if BaseClient is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "botocore is not installed. Install it with: pip install boto3"
        )

    return create_instructor(
        client,
        provider=Provider.BEDROCK,
        mode=mode,
        model=model,
        use_async=async_client,
        sync_types=(BaseClient,),
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

    try:
        import boto3
    except ImportError:
        raise ConfigurationError(
            "The boto3 package is required to use the AWS Bedrock provider. "
            "Install it with `pip install boto3`."
        ) from None
    aws_kwargs = {
        key: kwargs.pop(key, os.environ.get(key.upper()))
        for key in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token")
        if key in kwargs or key.upper() in os.environ
    }
    aws_kwargs["region_name"] = kwargs.pop(
        "region", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    selected_mode = mode or (
        Mode.TOOLS
        if "anthropic" in model_name.lower() or "claude" in model_name.lower()
        else Mode.MD_JSON
    )
    return from_bedrock(
        boto3.client("bedrock-runtime", **aws_kwargs),
        model=model_name,
        mode=selected_mode,
        async_client=async_client,
        **kwargs,
    )


__all__ = ["build_from_model", "from_bedrock"]
