"""Perplexity-specific utilities.

This module contains utilities specific to the Perplexity provider,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

from typing import Any

from ...mode import Mode
from ...utils.core import dump_message


def reask_perplexity_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Perplexity JSON mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user message requesting JSON correction)
    """
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}",
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def handle_perplexity_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle Perplexity JSON mode.

    Kwargs modifications:
    - Adds: "response_format" with json_schema
    """
    new_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {"schema": response_model.model_json_schema()},
    }

    return response_model, new_kwargs


# Handler registry for Perplexity
PERPLEXITY_HANDLERS = {
    Mode.PERPLEXITY_JSON: {
        "reask": reask_perplexity_json,
        "response": handle_perplexity_json,
    },
}
