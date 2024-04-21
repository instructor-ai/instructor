from __future__ import annotations

from typing import (
    Type,
    List,
    overload,
)

from ibm_watsonx_ai.foundation_models import ModelInference as Watsonx
from openai.types.chat import ChatCompletionMessageParam


import instructor

from .client import T

ALLOWED_MODES = {instructor.Mode.MD_JSON}


@overload
def from_watsonx(
    client: Watsonx,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs,
) -> instructor.Instructor: ...


@overload
def from_watsonx(
    client: Watsonx,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs,
) -> instructor.AsyncInstructor: ...


def from_watsonx(
    client: Watsonx,
    mode: instructor.Mode = instructor.Mode.MD_JSON,
    **kwargs,
) -> instructor.Instructor | instructor.AsyncInstructor:
    assert mode in ALLOWED_MODES, f"Mode must be one of {ALLOWED_MODES}"

    assert isinstance(
        client, Watsonx
    ), "Client must be an instance of ibm_watsonx_ai.foundation_models.ModelInference"

    def create(
        messages: List[ChatCompletionMessageParam],
        response_model: Type[T],
        validation_context: dict | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> T:
        """
        https://ibm.github.io/watsonx-ai-python-sdk/fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text_stream
        """

        return client.generate_text_stream(
            # TODO: improve naive implementation
            prompt="\n-----\n".join(
                f"""
            -----

            role: {m.role}

            content: {m.content}
            """
                for m in messages
            )
        )

    return instructor.Instructor(
        # TODO: add watsonx async
        # if client.async_mode:
        # return instructor.AsyncInstructor(
        # else:
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.WATSONX,
        mode=mode,
        **kwargs,
    )
