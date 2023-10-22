import enum
import functools
import inspect
import json
import logging

from typing import Any, Callable, List, Optional
import uuid
from pydantic import BaseModel, validate_call

from instructor import openai_schema


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
    assert (
        return_type != inspect.Signature.empty
    ), "Must have a return type hint that is a pydantic BaseModel"
    return inspect.isclass(return_type) and issubclass(return_type, BaseModel)


class Instructions:
    def __init__(
        self,
        name: str = None,
        id: str = None,
        log_handlers: List[logging.Handler] = None,
        finetune_format: FinetuneFormat = FinetuneFormat.MESSAGES,
        indent: int = 2,
    ):
        self.name = name
        self.id = id or str(uuid.uuid4())
        self.unique_id = str(uuid.uuid4())
        self.finetune_format = finetune_format
        self.indent = indent

        self.logger = logging.getLogger(self.name)
        for handler in log_handlers or []:
            self.logger.addHandler(handler)

    def distil(
        self,
        *args,
        name: str = None,
        mode: str = "distil",
        fine_tune_format: FinetuneFormat = None,
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

        if fine_tune_format is None:
            fine_tune_format = self.finetune_format

        def _wrap_distil(fn):
            msg = f"Return type hint for {fn} must subclass `pydantic.BaseModel'"
            assert is_return_type_base_model_or_instance(fn), msg

            @functools.wraps(fn)
            def _distil(*args, **kwargs):
                resp = fn(*args, **kwargs)
                self.track(
                    fn, args, kwargs, resp, name=name, finetune_format=fine_tune_format
                )

                return resp

            return _distil

        if len(args) == 1 and callable(args[0]):
            return _wrap_distil(args[0])

        return _wrap_distil

    @validate_call
    def track(
        self,
        fn: Callable[..., Any],
        args: tuple,
        kwargs: dict,
        resp: BaseModel,
        name: Optional[str] = None,
        finetune_format: FinetuneFormat = FinetuneFormat.MESSAGES,
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

        if finetune_format == FinetuneFormat.MESSAGES:
            openai_kwargs = self.openai_kwargs(name, fn, args, kwargs, base_model)
            openai_kwargs.append(
                {
                    "role": "assistant",
                    "function_call": {
                        "name": base_model.__name__,
                        "arguments": resp.model_dump_json(indent=self.indent),
                    },
                }
            )
            self.logger.info(json.dumps(openai_kwargs))

        if finetune_format == FinetuneFormat.RAW:
            function_body = dict(
                fn_name=name,
                fn_repr=format_function(fn),
                args=args,
                kwargs=kwargs,
                resp=resp.model_dump(),
                schema=base_model.model_json_schema(),
            )
            self.logger.info(json.dumps(function_body))

    def openai_kwargs(self, name, fn, args, kwargs, base_model):
        openai_function_call = openai_schema(base_model).openai_schema
        func_def = get_signature_from_fn(fn).replace(fn.__name__, name)

        str_args = ", ".join(map(str, args))
        str_kwargs = (
            ", ".join(f"{k}={json.dumps(v)}" for k, v in kwargs.items()) or None
        )
        call_args = ", ".join(filter(None, [str_args, str_kwargs]))

        function_body = {
            "messages": [
                {
                    "role": "system",
                    "content": f"Predict the results of this function:\n\n{func_def}",
                },
                {
                    "role": "user",
                    "content": f"Return `{name}({call_args})`",
                },
            ],
            "functions": [openai_function_call],
        }
        return function_body
