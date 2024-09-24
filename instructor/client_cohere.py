from __future__ import annotations

import cohere
import instructor
from functools import wraps
from typing import (
    TypeVar,
    overload,
)
from typing import Any
from typing_extensions import ParamSpec
from pydantic import BaseModel
from instructor.patch import handle_cohere_templating, handle_context
from instructor.process_response import handle_response_model
from instructor.retry import retry_async


T_Model = TypeVar("T_Model", bound=BaseModel)
T_ParamSpec = ParamSpec("T_ParamSpec")


@overload
def from_cohere(
    client: cohere.Client,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_cohere(
    client: cohere.AsyncClient,
    mode: instructor.Mode = instructor.Mode.COHERE_JSON_SCHEMA,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_cohere(
    client: cohere.Client | cohere.AsyncClient,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs: Any,
):
    assert mode in {
        instructor.Mode.COHERE_TOOLS,
        instructor.Mode.COHERE_JSON_SCHEMA,
    }, "Mode be one of {COHERE_TOOLS, COHERE_JSON_SCHEMA}"

    assert isinstance(
        client, (cohere.Client, cohere.AsyncClient)
    ), "Client must be an instance of cohere.Cohere or cohere.AsyncCohere"

    if isinstance(client, cohere.Client):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=client.chat, mode=mode),
            provider=instructor.Provider.COHERE,
            mode=mode,
            **kwargs,
        )

    @wraps(client.chat)
    async def new_create_async(
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,
        max_retries: int = 1,
        context: dict[str, Any] | None = None,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:
        prepared_response_model, new_kwargs = handle_response_model(
            response_model=response_model,
            mode=mode,
            **kwargs,
        )

        context = handle_context(context, validation_context)
        new_kwargs = handle_cohere_templating(new_kwargs, context)

        response = await retry_async(
            func=client.chat,
            response_model=prepared_response_model,
            context=context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=mode,
        )
        return response

    return instructor.AsyncInstructor(
        client=client,
        create=new_create_async,
        provider=instructor.Provider.COHERE,
        mode=mode,
        **kwargs,
    )
