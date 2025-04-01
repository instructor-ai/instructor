# type: ignore
import json
import logging
import re
from functools import wraps
from typing import Annotated, Any, Optional, TypeVar, cast
from docstring_parser import parse
from openai.types.chat import ChatCompletion
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    create_model,
)

from instructor.exceptions import IncompleteOutputException
from instructor.mode import Mode
from instructor.utils import (
    classproperty,
    extract_json_from_codeblock,
    map_to_gemini_function_schema,
)


T = TypeVar("T")
Model = TypeVar("Model", bound=BaseModel)

logger = logging.getLogger("instructor")

# No schema cache


# Utility functions for common JSON parsing operations
def _handle_incomplete_output(completion: Any) -> None:
    """Check if a completion was incomplete and raise appropriate exception."""
    if (
        hasattr(completion, "choices")
        and completion.choices[0].finish_reason == "length"
    ):
        raise IncompleteOutputException(last_completion=completion)

    # Handle Anthropic format
    if hasattr(completion, "stop_reason") and completion.stop_reason == "max_tokens":
        raise IncompleteOutputException(last_completion=completion)


def _extract_text_content(completion: Any) -> str:
    """Extract text content from various completion formats."""
    # OpenAI format
    if hasattr(completion, "choices"):
        return completion.choices[0].message.content or ""

    # Simple text format
    if hasattr(completion, "text"):
        return completion.text

    # Anthropic format
    if hasattr(completion, "content"):
        text_blocks = [c for c in completion.content if c.type == "text"]
        if text_blocks:
            return text_blocks[0].text

    # Bedrock format
    if isinstance(completion, dict) and "output" in completion:
        try:
            return completion.get("output").get("message").get("content")[0].get("text")
        except (AttributeError, IndexError):
            pass

    return ""


def _validate_model_from_json(
    cls: type[Model],
    json_str: str,
    validation_context: Optional[dict[str, Any]] = None,
    strict: Optional[bool] = None,
) -> Model:
    """Validate model from JSON string with appropriate error handling."""
    try:
        if strict:
            return cls.model_validate_json(
                json_str, context=validation_context, strict=True
            )
        else:
            # Allow control characters
            parsed = json.loads(json_str, strict=False)
            return cls.model_validate(parsed, context=validation_context, strict=False)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode error: {e}")
        raise ValueError(f"Failed to parse JSON: {e}") from e
    except Exception as e:
        logger.debug(f"Model validation error: {e}")
        raise


class OpenAISchema(BaseModel):
    # Ignore classproperty, since Pydantic doesn't understand it like it would a normal property.
    model_config = ConfigDict(ignored_types=(classproperty,))

    @classproperty
    def openai_schema(cls) -> dict[str, Any]:
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

    @classproperty
    def anthropic_schema(cls) -> dict[str, Any]:
        # Generate the Anthropic schema based on the OpenAI schema to avoid redundant schema generation
        openai_schema = cls.openai_schema
        return {
            "name": openai_schema["name"],
            "description": openai_schema["description"],
            "input_schema": cls.model_json_schema(),
        }

    @classproperty
    def gemini_schema(cls) -> Any:
        import google.generativeai.types as genai_types

        # Use OpenAI schema
        openai_schema = cls.openai_schema

        # Transform to Gemini format
        function = genai_types.FunctionDeclaration(
            name=openai_schema["name"],
            description=openai_schema["description"],
            parameters=map_to_gemini_function_schema(openai_schema["parameters"]),
        )

        return function

    @classmethod
    def from_response(
        cls,
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
        mode: Mode = Mode.TOOLS,
    ) -> BaseModel:
        """Execute the function from the response of an openai chat completion

        Parameters:
            completion (openai.ChatCompletion): The response from an openai chat completion
            throw_error (bool): Whether to throw an error if the function call is not detected
            context (dict): The context to use for validating the response
            strict (bool): Whether to use strict json parsing
            mode (Mode): The openai completion mode

        Returns:
            cls (OpenAISchema): An instance of the class
        """

        if mode == Mode.ANTHROPIC_TOOLS:
            return cls.parse_anthropic_tools(completion, validation_context, strict)

        if mode == Mode.ANTHROPIC_TOOLS or mode == Mode.ANTHROPIC_REASONING_TOOLS:
            return cls.parse_anthropic_tools(completion, validation_context, strict)

        if mode == Mode.ANTHROPIC_JSON:
            return cls.parse_anthropic_json(completion, validation_context, strict)

        if mode == Mode.BEDROCK_JSON:
            return cls.parse_bedrock_json(completion, validation_context, strict)

        if mode in {Mode.VERTEXAI_TOOLS, Mode.GEMINI_TOOLS}:
            return cls.parse_vertexai_tools(completion, validation_context)

        if mode == Mode.VERTEXAI_JSON:
            return cls.parse_vertexai_json(completion, validation_context, strict)

        if mode == Mode.COHERE_TOOLS:
            return cls.parse_cohere_tools(completion, validation_context, strict)

        if mode == Mode.GEMINI_JSON:
            return cls.parse_gemini_json(completion, validation_context, strict)

        if mode == Mode.GENAI_STRUCTURED_OUTPUTS:
            return cls.parse_genai_structured_outputs(
                completion, validation_context, strict
            )

        if mode == Mode.GEMINI_TOOLS:
            return cls.parse_gemini_tools(completion, validation_context, strict)

        if mode == Mode.GENAI_TOOLS:
            return cls.parse_genai_tools(completion, validation_context, strict)

        if mode == Mode.COHERE_JSON_SCHEMA:
            return cls.parse_cohere_json_schema(completion, validation_context, strict)

        if mode == Mode.WRITER_TOOLS:
            return cls.parse_writer_tools(completion, validation_context, strict)

        if not completion.choices:
            # This helps catch errors from OpenRouter
            if hasattr(completion, "error"):
                raise ValueError(completion.error)

            raise ValueError("No completion choices found")

        if completion.choices[0].finish_reason == "length":
            raise IncompleteOutputException(last_completion=completion)

        if mode == Mode.FUNCTIONS:
            Mode.warn_mode_functions_deprecation()
            return cls.parse_functions(completion, validation_context, strict)

        if mode == Mode.MISTRAL_STRUCTURED_OUTPUTS:
            return cls.parse_mistral_structured_outputs(
                completion, validation_context, strict
            )

        if mode in {
            Mode.TOOLS,
            Mode.MISTRAL_TOOLS,
            Mode.TOOLS_STRICT,
            Mode.CEREBRAS_TOOLS,
            Mode.FIREWORKS_TOOLS,
        }:
            return cls.parse_tools(completion, validation_context, strict)

        if mode in {
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.JSON_O1,
            Mode.CEREBRAS_JSON,
            Mode.FIREWORKS_JSON,
            Mode.PERPLEXITY_JSON,
            Mode.OPENROUTER_STRUCTURED_OUTPUTS,
        }:
            return cls.parse_json(completion, validation_context, strict)

        raise ValueError(f"Invalid patch mode: {mode}")

    @classmethod
    def parse_genai_structured_outputs(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        return cls.model_validate_json(
            completion.text, context=validation_context, strict=strict
        )

    @classmethod
    def parse_genai_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        from google.genai import types

        assert isinstance(completion, types.GenerateContentResponse)
        assert len(completion.candidates) == 1

        assert (
            len(completion.candidates[0].content.parts) == 1
        ), f"Instructor does not support multiple function calls, use List[Model] instead"
        function_call = completion.candidates[0].content.parts[0].function_call
        assert (
            function_call is not None
        ), f"Please return your response as a function call with the schema {cls.openai_schema} and the name {cls.openai_schema['name']}"

        assert function_call.name == cls.openai_schema["name"]
        return cls.model_validate(
            obj=function_call.args, context=validation_context, strict=strict
        )

    @classmethod
    def parse_cohere_json_schema(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ):
        assert hasattr(
            completion, "text"
        ), "Completion is not of type NonStreamedChatResponse"
        return cls.model_validate_json(
            completion.text, context=validation_context, strict=strict
        )

    @classmethod
    def parse_anthropic_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        from anthropic.types import Message

        if isinstance(completion, Message) and completion.stop_reason == "max_tokens":
            raise IncompleteOutputException(last_completion=completion)

        # Anthropic returns arguments as a dict, dump to json for model validation below
        tool_calls = [
            json.dumps(c.input) for c in completion.content if c.type == "tool_use"
        ]  # TODO update with anthropic specific types

        tool_calls_validator = TypeAdapter(
            Annotated[list[Any], Field(min_length=1, max_length=1)]
        )
        tool_call = tool_calls_validator.validate_python(tool_calls)[0]

        return cls.model_validate_json(
            tool_call, context=validation_context, strict=strict
        )

    @classmethod
    def parse_anthropic_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        from anthropic.types import Message

        if hasattr(completion, "choices"):
            completion = completion.choices[0]
            if completion.finish_reason == "length":
                raise IncompleteOutputException(last_completion=completion)
            text = completion.message.content
        else:
            assert isinstance(completion, Message)
            if completion.stop_reason == "max_tokens":
                raise IncompleteOutputException(last_completion=completion)
            # Find the first text block
            text_blocks = [c for c in completion.content if c.type == "text"]
            text = text_blocks[0].text

        extra_text = extract_json_from_codeblock(text)

        if strict:
            return cls.model_validate_json(
                extra_text, context=validation_context, strict=True
            )
        else:
            # Allow control characters.
            parsed = json.loads(extra_text, strict=False)
            # Pydantic non-strict: https://docs.pydantic.dev/latest/concepts/strict_mode/
            return cls.model_validate(parsed, context=validation_context, strict=False)

    @classmethod
    def parse_bedrock_json(
        cls: type[BaseModel],
        completion: Any,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        if isinstance(completion, dict):
            text = completion.get("output").get("message").get("content")[0].get("text")

            match = re.search(r"```?json(.*?)```?", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

            text = re.sub(r"```?json|\\n", "", text).strip()
        else:
            text = completion.text
        return cls.model_validate_json(text, context=validation_context, strict=strict)

    @classmethod
    def parse_gemini_json(
        cls: type[BaseModel],
        completion: Any,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        try:
            text = completion.text
        except ValueError:
            logger.debug(
                f"Error response: {completion.result.candidates[0].finish_reason}\n\n{completion.result.candidates[0].safety_ratings}"
            )

        try:
            extra_text = extract_json_from_codeblock(text)  # type: ignore
        except UnboundLocalError:
            raise ValueError("Unable to extract JSON from completion text") from None

        if strict:
            return cls.model_validate_json(
                extra_text, context=validation_context, strict=True
            )
        else:
            # Allow control characters.
            parsed = json.loads(extra_text, strict=False)
            # Pydantic non-strict: https://docs.pydantic.dev/latest/concepts/strict_mode/
            return cls.model_validate(parsed, context=validation_context, strict=False)

    @classmethod
    def parse_vertexai_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
    ) -> BaseModel:
        tool_call = completion.candidates[0].content.parts[0].function_call.args  # type: ignore
        model = {}
        for field in tool_call:  # type: ignore
            model[field] = tool_call[field]
        # We enable strict=False because the conversion from protobuf -> dict often results in types like ints being cast to floats, as a result in order for model.validate to work we need to disable strict mode.
        return cls.model_validate(model, context=validation_context, strict=False)

    @classmethod
    def parse_vertexai_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        return cls.model_validate_json(
            completion.text, context=validation_context, strict=strict
        )

    @classmethod
    def parse_cohere_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        text = cast(str, completion.text)  # type: ignore - TODO update with cohere specific types
        extra_text = extract_json_from_codeblock(text)
        return cls.model_validate_json(
            extra_text, context=validation_context, strict=strict
        )

    @classmethod
    def parse_writer_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message
        tool_calls = message.tool_calls
        assert (
            len(tool_calls) == 1
        ), "Instructor does not support multiple tool calls, use List[Model] instead"
        assert (
            tool_calls[0].function.name == cls.openai_schema["name"]
        ), "Tool name does not match"
        return cls.model_validate_json(
            tool_calls[0].function.arguments,
            context=validation_context,
            strict=strict,
        )

    @classmethod
    def parse_functions(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
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
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message
        # this field seems to be missing when using instructor with some other tools (e.g. litellm)
        # trying to fix this by adding a check

        if hasattr(message, "refusal"):
            assert (
                message.refusal is None
            ), f"Unable to generate a response due to {message.refusal}"
        assert (
            len(message.tool_calls or []) == 1
        ), f"Instructor does not support multiple tool calls, use List[Model] instead"
        tool_call = message.tool_calls[0]  # type: ignore
        assert (
            tool_call.function.name == cls.openai_schema["name"]  # type: ignore[index]
        ), "Tool name does not match"
        return cls.model_validate_json(
            tool_call.function.arguments,  # type: ignore
            context=validation_context,
            strict=strict,
        )

    @classmethod
    def parse_mistral_structured_outputs(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        if not completion.choices or len(completion.choices) > 1:
            raise ValueError(
                "Instructor does not support multiple tool calls, use list[Model] instead"
            )

        message = completion.choices[0].message

        return cls.model_validate_json(
            message.content, context=validation_context, strict=strict
        )

    @classmethod
    def parse_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        """Parse JSON mode responses using the optimized extraction and validation."""
        # Check for incomplete output
        _handle_incomplete_output(completion)

        # Extract text from the response
        message = _extract_text_content(completion)
        if not message:
            # Fallback for OpenAI format if _extract_text_content doesn't handle it
            message = completion.choices[0].message.content or ""

        # Extract JSON from the text
        json_content = extract_json_from_codeblock(message)

        # Validate the model from the JSON
        return _validate_model_from_json(cls, json_content, validation_context, strict)


def openai_schema(cls: type[BaseModel]) -> OpenAISchema:
    """
    Wrap a Pydantic model class to add OpenAISchema functionality.
    """
    if not issubclass(cls, BaseModel):
        raise TypeError("Class must be a subclass of pydantic.BaseModel")

    # Create the wrapped model
    schema = wraps(cls, updated=())(
        create_model(
            cls.__name__ if hasattr(cls, "__name__") else str(cls),
            __base__=(cls, OpenAISchema),
        )
    )

    return cast(OpenAISchema, schema)
