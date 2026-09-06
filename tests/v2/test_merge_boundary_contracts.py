"""Regression coverage for error paths introduced by the security merge."""

from __future__ import annotations

import base64
import socket
from typing import Any

import pytest
from pydantic import BaseModel
from urllib3.exceptions import NewConnectionError

from instructor.cache import make_request_cache_key
from instructor.v2.core.json import MAX_JSON_DEPTH, extract_json_from_codeblock
from instructor.v2.core.multimodal import PDF
from instructor.v2.core.remote import _PublicHTTPConnection
from instructor.v2.providers.anthropic.multimodal import pdf_to_anthropic
from instructor.v2.validation.async_validators import (
    async_field_validator,
    async_model_validator,
    run_async_validators,
)
from instructor.v2.core.errors import AsyncValidationError


class Answer(BaseModel):
    value: int


def test_cache_request_encodes_model_classes_without_losing_schema() -> None:
    options: dict[str, Any] = dict(
        args=(),
        response_model=Answer,
        provider="openai",
        mode="tools",
        namespace="test",
        context=None,
        strict=True,
    )
    key = make_request_cache_key(request={"model_type": Answer}, **options)
    assert key is not None
    assert key == (
        make_request_cache_key(
            request={"model_type": Answer.model_json_schema()}, **options
        )
    )


def test_json_depth_limit_rejects_parseable_nested_containers() -> None:
    nested = "[" * (MAX_JSON_DEPTH + 1) + "0" + "]" * (MAX_JSON_DEPTH + 1)
    with pytest.raises(ValueError, match="128 level limit"):
        extract_json_from_codeblock(nested)


def test_public_connection_closes_socket_when_source_bind_fails() -> None:
    # A numeric IPv4 destination with an IPv6 source fails before any connect.
    connection = _PublicHTTPConnection(
        "8.8.8.8",
        port=443,
        timeout=0.01,
        source_address=("::1", 0),
        socket_options=[],
    )
    with pytest.raises(NewConnectionError, match="Unable to connect") as caught:
        connection._new_conn()
    assert isinstance(caught.value.__cause__, OSError)
    assert isinstance(caught.value.__cause__.__cause__, socket.gaierror)
    assert connection.sock is None


def test_anthropic_uppercase_url_downloads_pdf_through_public_transport() -> None:
    source = "HTTPS://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    document = pdf_to_anthropic(PDF.model_construct(source=source, data=None))
    assert document["source"]["type"] == "base64"
    assert base64.b64decode(document["source"]["data"]).startswith(b"%PDF-")


@pytest.mark.asyncio
async def test_explicit_validator_runner_aggregates_nested_and_field_errors() -> None:
    class Rejected(BaseModel):
        value: int

        @async_field_validator("value")
        async def reject(cls, value: int) -> int:
            raise ValueError(f"Rejected {value}")

    class Parent(BaseModel):
        child: Rejected
        other: int

        @async_field_validator("other")
        async def reject_other(cls, value: int) -> int:
            raise ValueError(f"Other {value}")

    with pytest.raises(AsyncValidationError) as caught:
        await run_async_validators(
            Parent(child=Rejected(value=1), other=2), context=None
        )
    assert [str(error) for error in caught.value.errors] == ["Rejected 1", "Other 2"]


@pytest.mark.asyncio
async def test_explicit_validator_runner_accepts_unchanged_and_replaced_models() -> (
    None
):
    class Validated(BaseModel):
        value: int

        @async_field_validator("value")
        async def unchanged(cls, value: int) -> int:
            return value

        @async_model_validator()
        async def replace(self) -> Validated:
            return self.model_copy(update={"value": self.value + 1})

        @async_model_validator()
        async def no_replacement(self) -> None:
            return None

    original = Validated(value=1)
    validated = await run_async_validators(original, context=None)
    assert validated.value == 2
    assert original.value == 1
