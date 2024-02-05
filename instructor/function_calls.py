from typing import Any, Dict, Optional, Type, TypeVar
from docstring_parser import parse
from functools import wraps
from pydantic import BaseModel, create_model
from instructor.exceptions import IncompleteOutputException
import enum
import warnings

T = TypeVar("T")


class Mode(enum.Enum):
    """The mode to use for patching the client"""

    FUNCTIONS: str = "function_call"
    PARALLEL_TOOLS: str = "parallel_tool_call"
    TOOLS: str = "tool_call"
    JSON: str = "json_mode"
    MD_JSON: str = "markdown_json_mode"
    JSON_SCHEMA: str = "json_schema_mode"

    def __new__(cls, value: str) -> "Mode":
        member = object.__new__(cls)
        member._value_ = value

        # Deprecation warning for FUNCTIONS
        if value == "function_call":
            warnings.warn(
                "FUNCTIONS is deprecated and will be removed in future versions",
                DeprecationWarning,
                stacklevel=2,
            )

        return member


class OpenAISchema(BaseModel):  # type: ignore[misc]
    @classmethod  # type: ignore[misc]
    @property
    def openai_schema(cls) -> Dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        schema = cls.model_json_schema()
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
            ):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.__name__}` with all "
                    f"the required parameters with correct types"
                )

        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }

    @classmethod
    def from_response(
        cls,
        completion: T,
        validation_context: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
        mode: Mode = Mode.TOOLS,
    ) -> Dict[str, Any]:
        """Execute the function from the response of an openai chat completion

        Parameters:
            completion (openai.ChatCompletion): The response from an openai chat completion
            throw_error (bool): Whether to throw an error if the function call is not detected
            validation_context (dict): The validation context to use for validating the response
            strict (bool): Whether to use strict json parsing
            mode (Mode): The openai completion mode

        Returns:
            cls (OpenAISchema): An instance of the class
        """
        assert hasattr(completion, "choices")

        if completion.choices[0].finish_reason == "length":
            raise IncompleteOutputException()

        message = completion.choices[0].message

        if mode == Mode.FUNCTIONS:
            assert (
                message.function_call.name == cls.openai_schema["name"]  # type: ignore[index]
            ), "Function name does not match"
            return cls.model_validate_json(
                message.function_call.arguments,
                context=validation_context,
                strict=strict,
            )
        elif mode == Mode.TOOLS:
            assert (
                len(message.tool_calls) == 1
            ), "Instructor does not support multiple tool calls, use List[Model] instead."
            tool_call = message.tool_calls[0]
            assert (
                tool_call.function.name == cls.openai_schema["name"]  # type: ignore[index]
            ), "Tool name does not match"
            return cls.model_validate_json(
                tool_call.function.arguments,
                context=validation_context,
                strict=strict,
            )
        elif mode in {Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON}:
            return cls.model_validate_json(
                message.content,
                context=validation_context,
                strict=strict,
            )
        else:
            raise ValueError(f"Invalid patch mode: {mode}")


def openai_schema(cls: Type[BaseModel]) -> OpenAISchema:
    if not issubclass(cls, BaseModel):
        raise TypeError("Class must be a subclass of pydantic.BaseModel")

    return wraps(cls, updated=())(
        create_model(
            cls.__name__,
            __base__=(cls, OpenAISchema),
        )
    )
