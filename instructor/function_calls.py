from typing import Any, Dict, Optional, Type, TypeVar
from docstring_parser import parse
from functools import wraps
from pydantic import BaseModel, create_model
from openai.types.chat import ChatCompletion
from typing import Any, Dict, Optional, Type
from instructor.mode import Mode
from instructor.utils import extract_json_from_codeblock
from instructor.exceptions import IncompleteOutputException
from instructor.mode import Mode
import logging


T = TypeVar("T")

logger = logging.getLogger("instructor")


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
    @property
    def anthropic_schema(cls) -> Dict[str, Any]:
        return {
            "name": cls.openai_schema["name"],
            "description": cls.openai_schema["description"],
            "input_schema": cls.model_json_schema(),
        }

    @classmethod
    def from_response(
        cls,
        completion: ChatCompletion,
        validation_context: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
        mode: Mode = Mode.TOOLS,
    ) -> BaseModel:
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
        if mode == Mode.ANTHROPIC_TOOLS:
            return cls.parse_anthropic_tools(completion, validation_context, strict)

        if mode == Mode.ANTHROPIC_JSON:
            return cls.parse_anthropic_json(completion, validation_context, strict)

        if mode == Mode.COHERE_TOOLS:
            return cls.parse_cohere_tools(completion, validation_context, strict)

        if completion.choices[0].finish_reason == "length":
            raise IncompleteOutputException()

        if mode == Mode.FUNCTIONS:
            return cls.parse_functions(completion, validation_context, strict)

        if mode in {Mode.TOOLS, Mode.MISTRAL_TOOLS}:
            return cls.parse_tools(completion, validation_context, strict)

        if mode in {Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON}:
            return cls.parse_json(completion, validation_context, strict)

        raise ValueError(f"Invalid patch mode: {mode}")

    @classmethod
    def parse_anthropic_tools(
        cls: Type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        tool_call = [c.input for c in completion.content if c.type == "tool_use"][0]

        return cls.model_validate(tool_call, context=validation_context, strict=strict)  # type:ignore

    @classmethod
    def parse_anthropic_json(
        cls: Type[BaseModel],
        completion,
        validation_context: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        from anthropic.types import Message

        assert isinstance(completion, Message)

        text = completion.content[0].text
        extra_text = extract_json_from_codeblock(text)
        return cls.model_validate_json(
            extra_text, context=validation_context, strict=strict
        )

    @classmethod
    def parse_cohere_tools(
        cls: Type[BaseModel],
        completion,
        validation_context: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        text = completion.text
        extra_text = extract_json_from_codeblock(text)
        return cls.model_validate_json(
            extra_text, context=validation_context, strict=strict
        )

    @classmethod
    def parse_functions(
        cls: Type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message
        assert (
            message.function_call.name == cls.openai_schema["name"]  # type: ignore[index]
        ), "Function name does not match"
        return cls.model_validate_json(
            message.function_call.arguments,  # type: ignore[attr-defined]
            context=validation_context,
            strict=strict,
        )

    @classmethod
    def parse_tools(
        cls: Type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message
        assert (
            len(message.tool_calls or []) == 1
        ), "Instructor does not support multiple tool calls, use List[Model] instead."
        tool_call = message.tool_calls[0]  # type: ignore
        assert (
            tool_call.function.name == cls.openai_schema["name"]  # type: ignore[index]
        ), "Tool name does not match"
        return cls.model_validate_json(
            tool_call.function.arguments,
            context=validation_context,
            strict=strict,
        )

    @classmethod
    def parse_json(
        cls: Type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message.content or ""
        message = extract_json_from_codeblock(message)

        return cls.model_validate_json(
            message,
            context=validation_context,
            strict=strict,
        )


def openai_schema(cls: Type[BaseModel]) -> OpenAISchema:
    if not issubclass(cls, BaseModel):
        raise TypeError("Class must be a subclass of pydantic.BaseModel")

    return wraps(cls, updated=())(
        create_model(
            cls.__name__ if hasattr(cls, "__name__") else str(cls),
            __base__=(cls, OpenAISchema),
        )
    )  # type: ignore[all]
