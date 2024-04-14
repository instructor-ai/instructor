from __future__ import annotations

import cohere
import instructor
from functools import wraps
from typing import (
    Type,
    TypeVar,
    overload,
)
from typing_extensions import ParamSpec
from pydantic import BaseModel
from instructor.process_response import handle_response_model
from instructor.retry import retry_async


T_Model = TypeVar("T_Model", bound=BaseModel)
T_ParamSpec = ParamSpec("T_ParamSpec")


@overload
def from_cohere(
    client: cohere.Client,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs,
) -> instructor.Instructor: ...


@overload
def from_cohere(
    client: cohere.AsyncClient,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs,
) -> instructor.AsyncInstructor: ...


def from_cohere(
    client: cohere.Client | cohere.AsyncClient,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs,
):
    assert mode in {
        instructor.Mode.COHERE_TOOLS,
    }, "Mode be one of {instructor.Mode.COHERE_TOOLS}"

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
        response_model: Type[T_Model] = None,
        validation_context: dict = None,
        max_retries: int = 1,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:
        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )
        response = await retry_async(
            func=client.chat,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=mode,
        )  # type: ignore
        return response

    return instructor.AsyncInstructor(
        client=client,
        create=new_create_async,
        provider=instructor.Provider.COHERE,
        mode=mode,
        **kwargs,
    )
