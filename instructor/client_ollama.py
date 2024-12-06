from __future__ import annotations

from typing import Any, overload

import ollama

import instructor


@overload
def from_ollama(
    client: (
        ollama.Client
    ),
    mode: instructor.Mode = instructor.Mode.OLLAMA_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_ollama(
    client: (
       ollama.AsyncClient
    ),
    mode: instructor.Mode = instructor.Mode.OLLAMA_TOOLS,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_ollama(
    client: (
        ollama.Client
        | ollama.AsyncClient
    ),
    mode: instructor.Mode = instructor.Mode.OLLAMA_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    assert (
        mode
        in {
            instructor.Mode.OLLAMA_TOOLS,
        }
    ), "Mode be one of {instructor.Mode.OLLAMA_TOOLS}"

    assert isinstance(
        client,
        (
            ollama.Client,
            ollama.AsyncClient,
        ),
    ), "Client must be an instance of {ollama.Client, ollama.AsyncClient}"

    create = client.chat

    if isinstance(
        client,
        (
            ollama.Client,
        ),
    ):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.OLLAMA,
            mode=mode,
            **kwargs,
        )

    else:
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.OLLAMA,
            mode=mode,
            **kwargs,
        )