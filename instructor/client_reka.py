# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations

import reka
import instructor
from typing import overload, Optional
import inspect
from datetime import datetime
from openai.types.chat import ChatCompletion
import logging


class RekaClient:
    def __init__(self, api_key):
        reka.API_KEY = api_key


@overload
def from_reka(
    mode: instructor.Mode = instructor.Mode.MD_JSON,
    **kwargs,
) -> instructor.Instructor: ...


def from_reka(
    api_key: Optional[str] = None,
    mode: instructor.Mode = instructor.Mode.MD_JSON,
    model="reka-flash",
    **kwargs,
) -> instructor.Instructor | instructor.AsyncInstructor:
    client = RekaClient(api_key)
    assert mode in {
        instructor.Mode.MD_JSON,
    }, "Mode be one of {instructor.Mode.MD_JSON}"

    assert isinstance(
        client, (RekaClient, reka.chat)
    ), "Client must be an instance of reka.chat or reka.completion"
    assert not isinstance(
        client, (reka.AsyncRekaClient)
    ), "Reka does not support asynchronous clients"
    client.default_model = model
    return instructor.Instructor(
        client=client,
        create=instructor.patch(
            create=lambda **kw: reka_chat_wrapper(client.default_model, **kw), mode=mode
        ),
        provider=instructor.Provider.REKA,
        mode=mode,
        **kwargs,
    )


def reka_chat_wrapper(default_model, **kwargs):
    model = kwargs.pop("model", default_model)
    kwargs["model_name"] = model
    kwargs = reformat_openai_request_as_reka(kwargs)

    try:
        response = reka.chat(**kwargs)
        completion = reformat_reka_resp_as_chat_completion(
            response, kwargs.get("model_name", "reka-flash")
        )
        return completion
    except TypeError as e:
        logging.error(f"TypeError encountered: {e}")
        raise


def reformat_openai_request_as_reka(kwargs):
    messages = kwargs.pop("messages", [])
    conversation_history = kwargs.pop("conversation_history", [])

    # Process messages
    new_messages = []
    for msg in messages:
        msg_type = "model" if msg.get("role") == "assistant" else "human"
        text = (
            " ".join(str(part) for part in msg["content"])
            if isinstance(msg["content"], list)
            else msg["content"]
        )
        new_messages.append({"type": msg_type, "text": text})

    # Update conversation history
    if new_messages:
        conversation_history.append(new_messages[0])
        for current_msg in new_messages[1:]:
            last_msg = conversation_history[-1]
            if last_msg["type"] == current_msg["type"]:
                last_msg["text"] += "\n" + current_msg["text"]
            else:
                conversation_history.append(current_msg)
    kwargs["conversation_history"] = conversation_history

    # Adjust OpenAI-specific parameters for Reka API
    param_mapping = {
        "max_tokens": "request_output_length",
        "seed": "random_seed",
        "stop": "stop_words",
        "top_p": "runtime_top_p",
    }
    for openai_arg, reka_arg in param_mapping.items():
        if openai_arg in kwargs:
            kwargs[reka_arg] = kwargs.pop(openai_arg)

    # Validate against Reka's API signature
    chat_params = inspect.signature(reka.chat).parameters
    allowed_args = set(chat_params.keys())
    kwargs = {key: value for key, value in kwargs.items() if key in allowed_args}
    return kwargs


def reformat_reka_resp_as_chat_completion(response, model_name):
    finish_reason = response.get("finish_reason", "unknown")
    content = response.get("text", "")
    generated_tokens = response.get("metadata", {}).get("generated_tokens", 0)
    input_tokens = response.get("metadata", {}).get("input_tokens", 0)
    total_tokens = input_tokens + generated_tokens

    chat_completion_data = {
        "choices": [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "message": {"content": content, "role": "assistant"},
                "logprobs": None,
            }
        ],
        "created": int(datetime.now().timestamp()),
        "id": f"reka-{datetime.now().timestamp()}",
        "model": model_name or "reka-flash",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": generated_tokens,
            "prompt_tokens": input_tokens,
            "total_tokens": total_tokens,
        },
    }
    completion_instance = ChatCompletion(**chat_completion_data)
    return completion_instance
