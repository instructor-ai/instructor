# type: ignore
from __future__ import annotations

import google.generativeai as genai
import instructor

from typing import Any


def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    **kwargs: Any,
) -> instructor.Instructor:
    assert mode in {
        instructor.Mode.GEMINI_JSON,
        instructor.Mode.GEMINI_TOOLS,
    }, "Mode be one of {instructor.Mode.GEMINI_JSON, instructor.Mode.GEMINI_TOOLS}"

    assert isinstance(
        client,
        (genai.GenerativeModel),
    ), "Client must be an instance of genai.generativemodel"

    create = client.generate_content

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.GEMINI,
        mode=mode,
        **kwargs,
    )
