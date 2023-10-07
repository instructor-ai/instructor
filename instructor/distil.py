import enum
import functools
import inspect
import json

from typing import Any, Callable, Optional
from pydantic import BaseModel, validate_call

import inspect
import logging

from instructor import openai_schema

distil_logger = logging.getLogger("instructor.distil")


def logging(level=logging.INFO, handler=None, log_to_file=True, filename_prefix=None):
    """
    Configure the instructor module's logging.

    :param level: Log level.
    :param handler: Optional logging handler. If not provided, defaults to FileHandler or NullHandler based on log_to_file.
    :param log_to_file: If True and no handler is provided, logs to a file.
    :param filename: Optional filename for logging if log_to_file is True. Defaults to 'instructor.log'.
    """
    distil_logger.setLevel(level)

    # Clear existing handlers
    for h in distil_logger.handlers[:]:
        distil_logger.removeHandler(h)

    if handler:
        distil_logger.addHandler(handler)
    elif log_to_file:
        filename = filename_prefix or "instructor.log"
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        distil_logger.addHandler(file_handler)
    else:
        distil_logger.addHandler(logging.NullHandler())


class FinetuneFormat(enum.Enum):
    MESSAGES: str = "messages"
    RAW: str = "raw"


def get_signature_from_fn(fn: Callable) -> str:
    """
    Get the function signature as a string.

    :Example:

    >>> def my_function(a: int, b: int) -> int:
    >>>     return a + b
    >>>
    >>> get_signature_from_fn(my_function)
    "def my_function(a: int, b: int) -> int"

    :param fn: Function to get the signature for.
    :return: Function signature as a string.
    """
    sig = inspect.signature(fn)
    lines = f"def {fn.__name__}{sig}"
    docstring = inspect.getdoc(fn)
    if docstring:
        formatted_docstring = f'"""\n{docstring}\n"""'
    else:
        formatted_docstring = ""
    return f"{lines}\n{formatted_docstring}"


@functools.lru_cache()
def format_function(func: Callable) -> str:
    """
    Format a function as a string with docstring and body.
    """
    source_lines = inspect.getsourcelines(func)
    definition = " ".join(source_lines[0]).strip()

    docstring = inspect.getdoc(func)
    if docstring:
        formatted_docstring = f'"""\n{docstring}\n"""'
    else:
        formatted_docstring = ""

    body = inspect.getsource(func)
    body = body.replace(f"def {func.__name__}", "")

    return f"{definition}\n{formatted_docstring}\n{body}"


def is_return_type_base_model_or_instance(func: Callable[..., Any]) -> bool:
    """
    Check if the return type of a function is a pydantic BaseModel or an instance of it.

    :param func: Function to check.
    :return: True if the return type is a pydantic BaseModel or an instance of it.
    """
    return_type = inspect.signature(func).return_annotation
    return inspect.isclass(return_type) and issubclass(return_type, BaseModel)


@validate_call
def track(
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    resp: BaseModel,
    name: Optional[str] = None,
    finetune_format: FinetuneFormat = FinetuneFormat.RAW,
):
    """
    Track the function call and response in a log file, later used for finetuning.

    :param fn: Function to track.
    :param args: Arguments passed to the function.
    :param kwargs: Keyword arguments passed to the function.
    :param resp: Response returned by the function.
    :param name: Name of the function to track. Defaults to the function name.
    :param finetune_format: Format to use for finetuning. Defaults to "raw".
    """
    name = name if name else fn.__name__
    base_model: BaseModel = type(resp)

    if finetune_format == FinetuneFormat.RAW:
        function_body = dict(
            fn_name=name,
            fn_repr=format_function(fn),
            args=args,
            kwargs=kwargs,
            resp=resp.model_dump(),
            schema=base_model.model_json_schema(),
        )
        distil_logger.info(json.dumps(function_body))
        return

    if finetune_format == FinetuneFormat.MESSAGES:
        # This is the format that OpenAI's API expects for a finetune call
        openai_function_call = openai_schema(base_model).openai_schema
        function_definition = get_signature_from_fn(fn)
        function_body = {
            "messages": [
                {
                    "role": "system",
                    "content": f"Return the response from the function call.\n\n {function_definition}",
                },
                {
                    "role": "user",
                    "content": f"Return the results of the function with the following arguments:\n\n {name}(*{args}, **{kwargs})",
                },
                {
                    "role": "function",
                    "function_call": {
                        "name": openai_function_call["name"],
                        "augments": resp.model_dump(),
                    },
                },
            ],
            "functions": [openai_function_call],
            "function_call": {"name": name},
        }
        distil_logger.info(json.dumps(function_body))
        return
    raise ValueError(f"Invalid finetune format: {finetune_format}")


def distil(
    *args,
    name: str = None,
    mode: str = "distil",
    fine_tune_format: FinetuneFormat = FinetuneFormat.RAW,
):
    """
    Decorator to track the function call and response, supports distillation and dispatch modes.

    If used without arguments, it must be used as a decorator.

    :Example:

    >>> @distil
    >>> def my_function() -> MyModel:
    >>>     return MyModel()
    >>>
    >>> @distil(name="my_function")
    >>> def my_function() -> MyModel:
    >>>     return MyModel()

    :param fn: Function to track.
    :param name: Name of the function to track. Defaults to the function name.
    :param mode: Mode to use for distillation. Defaults to "distil".
    """
    allowed_modes = {"distil", "dispatch"}
    assert mode in allowed_modes, f"Must be in {allowed_modes}"
    assert mode == "distil", "Only distil mode is supported at the moment."

    def _wrap_distil(fn):
        msg = f"Return type hint for {fn} must subclass `pydantic.BaseModel'"
        assert is_return_type_base_model_or_instance(fn), msg

        @functools.wraps(fn)
        def _distil(*args, **kwargs):
            resp = fn(*args, **kwargs)
            track(fn, args, kwargs, resp, name=name, finetune_format=fine_tune_format)
            return resp

        return _distil

    if len(args) == 1 and callable(args[0]):
        return _wrap_distil(args[0])

    return _wrap_distil
